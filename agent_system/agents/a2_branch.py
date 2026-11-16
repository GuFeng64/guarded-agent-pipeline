# -*- coding: utf-8 -*-
"""agents/a2_branch.py — 参数化分支 Agent（读缓存回放）

## ⚠️ 重要限定（必须写进论文，章程规则 Z2）
    本阶段分支 Agent 读取**已缓存的 OOF 预测**，不是实时推理。
    ⇒ 并行的真实收益来自"编排与容错"，**不是推理加速**。

## 单位与折结构（关键，踩过两次坑）
    ① 缓存中的分支预测**已是归一化空间**（相对该折 mu/sd）⇒ 不可再除 mu/sd
    ② 融合的仿射是**在每折的训练行上单独拟合**的，
       且用的是该折模型的**内层预测 `f{fi}_*_tr`**，
       **不是**"全局 OOF 在训练行上的值" ⇒ 必须按折提供输出

## 接口
    agent.fold_train(fi) -> (m_tr, nt)   该折训练行的预测（内层）
    agent.fold_test(fi)  -> (m_te, nt)   该折测试行的预测（OOF）
    agent.oof(n, nt)     -> (n, nt)      全局 OOF（仅用于评估/报告）

## 自测
    python a2_branch.py
"""
from __future__ import annotations

import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_AG = os.path.dirname(_HERE)
_REV = os.path.dirname(_AG)
_EXP = os.path.join(_REV, 'experiments')
_CODE = os.path.join(_REV, 'code')
for _p in (_REV, _CODE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

__all__ = ['BRANCH_SPEC', 'BranchAgent', 'load_folds', 'build_all_oof']

BRANCH_SPEC = {
    'b1':  {'file': 'step426_4v_vi4_cache.npz',
            'tr': 'f{fi}_b1_tr', 'te': 'f{fi}_b1_te',
            'col': None, 'squeeze': True},
    'CNN': {'file': 'step423_v37_cnn_cache.npz',
            'tr': 'f{fi}_cnn_tr', 'te': 'f{fi}_cnn_te',
            'col': None, 'squeeze': True},
    'b7':  {'file': 'step299_allbranches_cache.npz',
            'tr': 'f{fi}_cand_tr', 'te': 'f{fi}_cand_te',
            'col': 4, 'squeeze': False},
    'b8':  {'file': 'step299_allbranches_cache.npz',
            'tr': 'f{fi}_cand_tr', 'te': 'f{fi}_cand_te',
            'col': 5, 'squeeze': False},
    'b9':  {'file': 'step299_allbranches_cache.npz',
            'tr': 'f{fi}_cand_tr', 'te': 'f{fi}_cand_te',
            'col': 6, 'squeeze': False},
}


def load_folds():
    """按牛分组 5 折（与 v39 定稿完全相同的划分）。"""
    from sklearn.model_selection import GroupKFold
    import mems_data as M
    d = M.load()
    Y = np.asarray(d['Y'], dtype=float)
    cow = np.asarray(d['cow'])
    n = len(Y)
    folds = list(GroupKFold(5).split(np.zeros((n, 1)), Y[:, 0], cow))
    return Y, cow, folds


class BranchAgent:
    """一个分支 = 一个 Agent。只读自己的缓存文件（上下文隔离的体现）。"""

    def __init__(self, name: str, folds, exp_dir: str = _EXP):
        if name not in BRANCH_SPEC:
            raise KeyError(f'未知分支 {name!r}；已知 {sorted(BRANCH_SPEC)}')
        self.name = name
        self.spec = BRANCH_SPEC[name]
        self.folds = folds
        self.exp_dir = exp_dir
        self._cache = None

    @property
    def cache(self):
        if self._cache is None:
            self._cache = np.load(os.path.join(self.exp_dir, self.spec['file']),
                                  allow_pickle=True)
        return self._cache

    def _slice(self, arr):
        if self.spec['col'] is not None:
            return arr[..., self.spec['col']]
        if self.spec['squeeze']:
            return arr[:, :, 0]
        return arr

    def fold_train(self, fi: int) -> np.ndarray:
        key = self.spec['tr'].format(fi=fi)
        v = self._slice(np.asarray(self.cache[key], dtype=float))
        assert v.shape[0] == len(self.folds[fi][0]), (
            f'{self.name} fold{fi} train: 行数 {v.shape[0]} ≠ {len(self.folds[fi][0])}')
        return v

    def fold_test(self, fi: int) -> np.ndarray:
        key = self.spec['te'].format(fi=fi)
        v = self._slice(np.asarray(self.cache[key], dtype=float))
        assert v.shape[0] == len(self.folds[fi][1]), (
            f'{self.name} fold{fi} test: 行数 {v.shape[0]} ≠ {len(self.folds[fi][1])}')
        return v

    def oof(self, n: int, n_targets: int) -> np.ndarray:
        """全局 OOF（仅用于评估/报告，**不用于融合拟合**）。"""
        out = np.full((n, n_targets), np.nan)
        for fi, (tr, te) in enumerate(self.folds):
            v = self.fold_test(fi)
            assert v.shape[1] == n_targets
            out[te] = v
        assert not np.isnan(out).any(), f'{self.name}: OOF 有缺失行'
        return out

    def __repr__(self):
        return f'<BranchAgent {self.name} file={self.spec["file"]}>'


def build_all_oof(names=None, exp_dir: str = _EXP):
    Y, cow, folds = load_folds()
    n, nt = Y.shape
    names = list(names or BRANCH_SPEC)
    return {nm: BranchAgent(nm, folds, exp_dir).oof(n, nt) for nm in names}, Y, cow, folds


# ---------------------------------------------------------------- 融合（与 v39 一致）

ORDER = ('b1', 'CNN', 'b7', 'b8', 'b9')


def fuse_equal(agents, folds, Y, mu, sd):
    """等权 + 逐目标仿射；**每折用该折的 fold_train 拟合**（与 v39 同）。"""
    n, nt = Y.shape
    PRED = np.zeros((n, nt))
    for fi, (tr, te) in enumerate(folds):
        Ptr = np.stack([agents[nm].fold_train(fi) for nm in ORDER], -1)   # (m, nt, 5)
        Pte = np.stack([agents[nm].fold_test(fi) for nm in ORDER], -1)
        for ti in range(nt):
            y = (Y[tr, ti] - mu[fi][ti]) / sd[fi][ti]
            w = np.ones(len(ORDER)) / len(ORDER)
            A = np.vstack([Ptr[:, ti, :] @ w, np.ones(len(tr))]).T
            cf, *_ = np.linalg.lstsq(A, y, rcond=None)
            PRED[te, ti] = (cf[0] * (Pte[:, ti, :] @ w) + cf[1]) * sd[fi][ti] + mu[fi][ti]
    return PRED


# ---------------------------------------------------------------- 自测

def _self_test() -> int:
    ok, fail = 0, []
    Y, cow, folds = load_folds()
    n, nt = Y.shape
    print('=' * 72)
    print('a2_branch.py 自测')
    print('=' * 72)
    print(f'  n={n} nt={nt} folds={[len(te) for _, te in folds]}')

    agents = {nm: BranchAgent(nm, folds) for nm in BRANCH_SPEC}

    # ① 五个分支的 fold_train / fold_test 形状正确
    for nm, ag in agents.items():
        good = all(ag.fold_train(fi).shape == (len(tr), nt)
                   and ag.fold_test(fi).shape == (len(te), nt)
                   for fi, (tr, te) in enumerate(folds))
        if good:
            ok += 1
        else:
            fail.append(f'{nm}: fold_train/fold_test 形状异常')

    # ② OOF 行必须来自对应测试折
    for nm, ag in agents.items():
        p = ag.oof(n, nt)
        if all(np.array_equal(p[te], ag.fold_test(fi))
               for fi, (tr, te) in enumerate(folds)):
            ok += 1
        else:
            fail.append(f'{nm}: OOF 行与 fold_test 不符')

    # ③ 融合复现 gap = 0.0329（不含 TS 闭合）
    B = np.load(os.path.join(_EXP, 'step426_4v_vi4_cache.npz'), allow_pickle=True)
    MU = [B[f'f{fi}_mu'] for fi in range(5)]
    SD = [B[f'f{fi}_sd'] for fi in range(5)]
    PRED = fuse_equal(agents, folds, Y, MU, SD)
    import metrics as MT
    Ti = np.concatenate([te for _, te in folds])
    r2 = MT.r2_by_target(Y, PRED, Ti)
    gap = float(sum(0.98 - r2[t] for t in MT.MAIN3))
    # ⭐ 等权融合本身 = 0.0329；v39 定稿的 0.0324 是**再叠加 TS 结构性闭合后**的值
    print(f'  等权融合 gap = {gap:.4f}（期望 0.0329，不含 TS 闭合）')
    if abs(gap - 0.0329) < 1e-3:
        ok += 1
    else:
        fail.append(f'融合 gap={gap:.4f}，期望 0.0329（不含 TS 闭合）')

    # ④ 与 step502 的 PRED 逐元素一致
    Z = np.load(os.path.join(_EXP, 'step502_fix.npz'), allow_pickle=True)
    if np.allclose(PRED, Z['PRED'], atol=1e-10):
        ok += 1
    else:
        fail.append(f'与 step502 的 PRED 不一致（max diff '
                    f'{np.abs(PRED - Z["PRED"]).max():.2e}）')

    print(f'  通过 {ok} 项')
    if fail:
        print(f'  ❌ 失败 {len(fail)} 项：')
        for f_ in fail:
            print('     -', f_)
        return 1
    print('  ✅ 全部通过：按折输出正确，等权融合 gap=0.0329，'
          '与 step502 的 PRED 逐元素一致（TS 闭合另由 a3_aggregate 施加）')
    return 0


if __name__ == '__main__':
    sys.exit(_self_test())
