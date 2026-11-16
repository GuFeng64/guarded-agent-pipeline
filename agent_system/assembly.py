# -*- coding: utf-8 -*-
"""assembly.py — T1.2 并行装配：5 个分支 Agent 并发推理后按规范顺序归并

## 判据（AGENT_TASKS T1.2）
    并行装配的结果必须与串行**逐元素一致**，且融合 gap 复现 0.0329。

## 实现
    · 用 BranchAgent 作为每个分支的 Agent（读缓存回放）
    · 用 Orchestrator.run_branches 并发调度，带超时与结构化失败
    · 用 Orchestrator.assemble 按 CANONICAL_BRANCHES 顺序归并
    · 融合：等权 + 逐目标仿射（每折在 fold_train 上拟合）

## 自测
    python assembly.py
"""
from __future__ import annotations

import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_AGENTS = os.path.join(_HERE, 'agents')
for _p in (_HERE, _AGENTS, _HERE.replace(os.sep + 'agent_system', '') + os.sep + 'code'):
    if _p not in sys.path:
        sys.path.insert(0, _p)

__all__ = ['assemble_parallel', 'assemble_serial', 'fuse']

ORDER = ('b1', 'CNN', 'b7', 'b8', 'b9')


def _load():
    from sklearn.model_selection import GroupKFold
    import mems_data as M
    d = M.load()
    Y = np.asarray(d['Y'], dtype=float)
    cow = np.asarray(d['cow'])
    n = len(Y)
    folds = list(GroupKFold(5).split(np.zeros((n, 1)), Y[:, 0], cow))
    B = np.load(os.path.join(os.path.dirname(_HERE), 'experiments',
                             'step426_4v_vi4_cache.npz'), allow_pickle=True)
    MU = [B[f'f{fi}_mu'] for fi in range(5)]
    SD = [B[f'f{fi}_sd'] for fi in range(5)]
    return Y, cow, folds, MU, SD


def assemble_serial(Y, cow, folds):
    """串行：逐个分支取 fold_train / fold_test。"""
    from agents.a2_branch import BranchAgent
    return {nm: (BranchAgent(nm, folds).fold_train(0),
                 BranchAgent(nm, folds).fold_test(0)) for nm in ORDER}


def assemble_parallel(folds, timeout=30.0):
    """并行：5 个分支 Agent 并发，返回 {name: {'tr': [..], 'te': [..]}} 与编排日志。"""
    from orchestrator import Orchestrator, CANONICAL_BRANCHES
    from agents.a2_branch import BranchAgent

    agents = {nm: BranchAgent(nm, folds) for nm in CANONICAL_BRANCHES}

    def call(nm):
        ag = agents[nm]
        return {'tr': [ag.fold_train(fi) for fi in range(5)],
                'te': [ag.fold_test(fi) for fi in range(5)]}

    orch = Orchestrator(timeout=timeout, max_workers=5)
    res = orch.run_branches(list(CANONICAL_BRANCHES), call)
    kept, degraded, missing = orch.assemble(res, allow_degraded=True)
    out = {nm: res[nm].pred for nm in CANONICAL_BRANCHES if res[nm].ok}
    return out, degraded, missing, orch


def fuse(preds, folds, Y, mu, sd, order=None):
    """等权 + 逐目标仿射；preds[nm] = {'tr': [...], 'te': [...]}"""
    order = list(order or [n for n in ORDER if n in preds])
    n, nt = Y.shape
    PRED = np.zeros((n, nt))
    for fi, (tr, te) in enumerate(folds):
        Ptr = np.stack([preds[nm]['tr'][fi] for nm in order], -1)
        Pte = np.stack([preds[nm]['te'][fi] for nm in order], -1)
        for ti in range(nt):
            y = (Y[tr, ti] - mu[fi][ti]) / sd[fi][ti]
            w = np.ones(len(order)) / len(order)
            A = np.vstack([Ptr[:, ti, :] @ w, np.ones(len(tr))]).T
            cf, *_ = np.linalg.lstsq(A, y, rcond=None)
            PRED[te, ti] = ((cf[0] * (Pte[:, ti, :] @ w) + cf[1])
                            * sd[fi][ti] + mu[fi][ti])
    return PRED


def _self_test() -> int:
    ok, fail = 0, []
    print('=' * 72)
    print('assembly.py 自测（并行装配 vs 串行）')
    print('=' * 72)

    Y, cow, folds, MU, SD = _load()
    n, nt = Y.shape
    print(f'  n={n} nt={nt}')

    # ---- 串行 ----
    ser = {}
    from agents.a2_branch import BranchAgent
    for nm in ORDER:
        ag = BranchAgent(nm, folds)
        ser[nm] = {'tr': [ag.fold_train(fi) for fi in range(5)],
                   'te': [ag.fold_test(fi) for fi in range(5)]}

    # ---- 并行 ----
    par, degraded, missing, orch = assemble_parallel(folds)

    # ① 并行与串行逐元素一致
    all_same = True
    for nm in ORDER:
        for fi in range(5):
            if not (np.array_equal(par[nm]['tr'][fi], ser[nm]['tr'][fi])
                    and np.array_equal(par[nm]['te'][fi], ser[nm]['te'][fi])):
                all_same = False
    if all_same:
        ok += 1
    else:
        fail.append('并行与串行的分支输出不一致')

    # ② 无缺失、未降级
    if (not degraded) and (not missing) and len(par) == 5:
        ok += 1
    else:
        fail.append(f'degraded={degraded} missing={missing}')

    # ③ 并行装配的融合 gap 复现 0.0329
    import metrics as MT
    Ti = np.concatenate([te for _, te in folds])
    P_par = fuse(par, folds, Y, MU, SD)
    r2 = MT.r2_by_target(Y, P_par, Ti)
    gap = float(sum(0.98 - r2[t] for t in MT.MAIN3))
    print(f'  并行装配融合 gap = {gap:.4f}（期望 0.0329）')
    if abs(gap - 0.0329) < 1e-3:
        ok += 1
    else:
        fail.append(f'并行装配 gap={gap:.4f}，期望 0.0329')

    # ④ 并行融合 == 串行融合（逐元素）
    P_ser = fuse(ser, folds, Y, MU, SD)
    if np.allclose(P_par, P_ser, atol=1e-10):
        ok += 1
    else:
        fail.append(f'并行/串行融合不一致 max={np.abs(P_par - P_ser).max():.2e}')

    # ⑤ 注入 b8 超时 ⇒ 降级标志正确，且退化量与表一致
    from orchestrator import Orchestrator
    from agents.a2_branch import BranchAgent as BA
    import time

    def slow(nm):
        if nm == 'b8':
            time.sleep(2.0)
        ag = BA(nm, folds)
        return {'tr': [ag.fold_train(fi) for fi in range(5)],
                'te': [ag.fold_test(fi) for fi in range(5)]}

    orch2 = Orchestrator(timeout=0.5, max_workers=5)
    res2 = orch2.run_branches(list(ORDER), slow)
    kept2, deg2, miss2 = orch2.assemble(res2)
    if deg2 and miss2 == ['b8']:
        ok += 1
    else:
        fail.append(f'超时降级标志错误 deg={deg2} miss={miss2}')

    # ⑥ 降级后 gap 与退化表一致（+0.0022）
    par2 = {nm: res2[nm].pred for nm in ORDER if res2[nm].ok}
    P_deg = fuse(par2, folds, Y, MU, SD)
    r2d = MT.r2_by_target(Y, P_deg, Ti)
    gap_d = float(sum(0.98 - r2d[t] for t in MT.MAIN3))
    from degradation import DEGRADATION_TABLE
    exp = DEGRADATION_TABLE['b8']
    print(f'  丢 b8 后 gap = {gap_d:.4f}（Δ={gap_d - gap:+.4f}，表值 {exp:.4f}）')
    if abs((gap_d - gap) - exp) < 8e-4:
        ok += 1
    else:
        fail.append(f'丢 b8 的退化量不符：实测 {gap_d - gap:+.4f} vs 表 {exp:.4f}')

    print(f'  通过 {ok} 项')
    if fail:
        print(f'  ❌ 失败 {len(fail)} 项：')
        for f_ in fail:
            print('     -', f_)
        return 1
    print('  ✅ 全部通过：并行≡串行；gap=0.0329；超时降级且退化量与表一致')
    return 0


if __name__ == '__main__':
    sys.exit(_self_test())
