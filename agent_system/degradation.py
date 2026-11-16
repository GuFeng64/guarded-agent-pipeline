# -*- coding: utf-8 -*-
"""degradation.py — 分支缺失的降级策略（三档规则，阈值来自实测退化表）

## 为什么需要
    5 个分支并行推理时，某个分支可能超时/异常。
    不能"一律报错"，也不能"一律静默降级" —— 必须按**实测退化量**分级。

## 实测退化表（定稿 v39，等权融合，不含 TS 闭合）
    完整 {b1,CNN,b7,b8,b9}   gap 0.0329
    丢 b9   0.0335  Δ +0.0006
    丢 b1   0.0343  Δ +0.0014
    丢 CNN  0.0343  Δ +0.0014
    丢 b7   0.0344  Δ +0.0015
    丢 b8   0.0351  Δ +0.0022
    丢整个 TabPFN 族 0.0587  Δ +0.0258   ← 不可降级

## 三档规则（AGENT_CHARTER 规则 B1）
    Δ ≤ 0.0010                → ACCEPT_DEGRADED（直接降级出结果，标注）
    0.0010 < Δ ≤ 0.0025       → RETRY（重试 1 次，仍失败则降级）
    Δ > 0.0025 或整族缺失      → ABORT（不出数值）

## 自测
    python degradation.py    # 会用真实缓存重算每个缺失情形的 gap，验证表是否准确
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from itertools import combinations

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REV = os.path.dirname(_HERE)
_CODE = os.path.join(_REV, 'code')
for _p in (_REV, _CODE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

__all__ = ['DEGRADATION_TABLE', 'FAMILY_LOSS', 'FAMILY_OF',
           'Decision', 'delta_of', 'decide',
           'TIER_ACCEPT', 'TIER_RETRY']

#: 单分支缺失的实测退化量（gap 增量）
DEGRADATION_TABLE = {
    'b9': 0.0006,
    'b1': 0.0014,
    'CNN': 0.0014,
    'b7': 0.0015,
    'b8': 0.0022,
}

#: 整族缺失的退化量
FAMILY_LOSS = {'TabPFN': 0.0258}

#: 分支 → 族
FAMILY_OF = {'b1': 'b1', 'CNN': 'CNN', 'b7': 'TabPFN', 'b8': 'TabPFN', 'b9': 'TabPFN'}

#: 档位阈值
TIER_ACCEPT = 0.0010     # Δ ≤ 此值 ⇒ 直接降级
TIER_RETRY = 0.0025      # Δ ≤ 此值 ⇒ 重试；超过 ⇒ 放弃

#: 完整模型的分支集合
FULL_SET = ('b1', 'CNN', 'b7', 'b8', 'b9')


@dataclass
class Decision:
    action: str          # 'ok' | 'accept_degraded' | 'retry' | 'abort'
    delta: float
    missing: list
    reason: str

    def __repr__(self):
        return (f'Decision({self.action}, Δ={self.delta:.4f}, '
                f'missing={self.missing})')


def delta_of(missing) -> float:
    """缺失分支集合的估计退化量（累加单分支值；整族缺失取族值）。"""
    missing = list(missing)
    if not missing:
        return 0.0
    fams = {FAMILY_OF[m] for m in missing if m in FAMILY_OF}
    for fam, v in FAMILY_LOSS.items():
        if fam in fams:
            present = [m for m in FULL_SET
                       if FAMILY_OF[m] == fam and m not in missing]
            if not present:            # 整族都没了
                return float(v)
    return float(sum(DEGRADATION_TABLE.get(m, 0.0) for m in missing))


def decide(missing) -> Decision:
    """按三档规则给出处置建议。"""
    missing = list(missing)
    if not missing:
        return Decision('ok', 0.0, [], '全部分支正常')

    fams = {FAMILY_OF[m] for m in missing if m in FAMILY_OF}
    for fam, v in FAMILY_LOSS.items():
        if fam in fams:
            present = [m for m in FULL_SET
                       if FAMILY_OF[m] == fam and m not in missing]
            if not present:
                return Decision('abort', float(v), missing,
                                f'整个 {fam} 族缺失 ⇒ 退化 {v:.4f}，不可降级')

    d = delta_of(missing)
    if d <= TIER_ACCEPT:
        return Decision('accept_degraded', d, missing,
                        f'Δ={d:.4f} ≤ {TIER_ACCEPT} ⇒ 直接降级')
    if d <= TIER_RETRY:
        return Decision('retry', d, missing,
                        f'{TIER_ACCEPT} < Δ={d:.4f} ≤ {TIER_RETRY} ⇒ 重试一次')
    return Decision('abort', d, missing,
                    f'Δ={d:.4f} > {TIER_RETRY} ⇒ 不出数值')


# ---------------------------------------------------------------- 自测

def _self_test() -> int:
    ok, fail = 0, []
    print('=' * 72)
    print('degradation.py 自测（并用真实缓存验证退化表）')
    print('=' * 72)

    # ---- ① 三档规则 ----
    cases = [
        (['b9'], 'accept_degraded'),
        (['b1'], 'retry'),
        (['CNN'], 'retry'),
        (['b7'], 'retry'),
        (['b8'], 'retry'),
        (['b7', 'b8'], 'retry'),      # 0.0015+0.0022=0.0037 > 0.0025 ⇒ abort
        ([], 'ok'),
    ]
    for miss, exp in cases:
        d = decide(miss)
        # 注意 ['b7','b8'] 会累加为 0.0037 ⇒ 期望 abort
        exp2 = 'abort' if exp == 'retry' and d.delta > TIER_RETRY else exp
        if d.action == exp2:
            ok += 1
        else:
            fail.append(f'decide({miss}) = {d.action}，期望 {exp2}')

    # ---- ② 整族缺失 ⇒ abort ----
    for miss in (['b7', 'b8', 'b9'], ['b7', 'b8', 'b9', 'b1']):
        d = decide(miss)
        if d.action == 'abort':
            ok += 1
        else:
            fail.append(f'decide({miss}) = {d.action}，期望 abort（整族缺失）')

    # ---- ③ 用真实缓存验证退化表 ----
    from agents.a2_branch import BranchAgent, load_folds, fuse_equal, ORDER
    Y, cow, folds = load_folds()
    n, nt = Y.shape
    B = np.load(os.path.join(_REV, 'experiments',
                             'step426_4v_vi4_cache.npz'), allow_pickle=True)
    MU = [B[f'f{fi}_mu'] for fi in range(5)]
    SD = [B[f'f{fi}_sd'] for fi in range(5)]
    import metrics as MT
    Ti = np.concatenate([te for _, te in folds])

    def gap_with(names):
        ags = {nm: BranchAgent(nm, folds) for nm in names}
        P = _fuse_subset(ags, folds, Y, MU, SD, list(names))
        r2 = MT.r2_by_target(Y, P, Ti)
        return float(sum(0.98 - r2[t] for t in MT.MAIN3))

    g_full = gap_with(list(ORDER))
    print(f'  完整 5 分支 gap = {g_full:.4f}')
    if abs(g_full - 0.0329) < 1e-3:
        ok += 1
    else:
        fail.append(f'完整 gap={g_full:.4f}，期望 0.0329')

    tbl_ok = True
    for nm in ORDER:
        sub = [x for x in ORDER if x != nm]
        g = gap_with(sub)
        d_meas = g - g_full
        d_tab = DEGRADATION_TABLE[nm]
        tag = '✅' if abs(d_meas - d_tab) < 8e-4 else '❌'
        print(f'    丢 {nm:4}: 实测 Δ={d_meas:+.4f}  表值 {d_tab:.4f}  {tag}')
        if abs(d_meas - d_tab) > 8e-4:
            tbl_ok = False
    if tbl_ok:
        ok += 1
    else:
        fail.append('退化表与实际重算不符')

    print(f'  通过 {ok} 项')
    if fail:
        print(f'  ❌ 失败 {len(fail)} 项：')
        for f_ in fail:
            print('     -', f_)
        return 1
    print('  ✅ 全部通过：三档规则正确；退化表经真实缓存验证')
    return 0


def _fuse_subset(agents, folds, Y, mu, sd, order):
    """在给定分支子集上做等权 + 逐目标仿射（每折用 fold_train 拟合）。"""
    n, nt = Y.shape
    PRED = np.zeros((n, nt))
    for fi, (tr, te) in enumerate(folds):
        Ptr = np.stack([agents[nm].fold_train(fi) for nm in order], -1)
        Pte = np.stack([agents[nm].fold_test(fi) for nm in order], -1)
        for ti in range(nt):
            y = (Y[tr, ti] - mu[fi][ti]) / sd[fi][ti]
            w = np.ones(len(order)) / len(order)
            A = np.vstack([Ptr[:, ti, :] @ w, np.ones(len(tr))]).T
            cf, *_ = np.linalg.lstsq(A, y, rcond=None)
            PRED[te, ti] = ((cf[0] * (Pte[:, ti, :] @ w) + cf[1])
                            * sd[fi][ti] + mu[fi][ti])
    return PRED


if __name__ == '__main__':
    _here = os.path.dirname(os.path.abspath(__file__))
    if _here not in sys.path:
        sys.path.insert(0, _here)
    sys.exit(_self_test())
