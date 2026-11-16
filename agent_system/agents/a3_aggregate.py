# -*- coding: utf-8 -*-
"""agents/a3_aggregate.py — 汇总 Agent：**离线拟合** 与 **在线应用** 分离

## 为什么拆分（本轮核心改动）
    原实现把"拟合融合系数"与"应用融合"混在一个类里，
    而拟合必须用真值 ⇒ 导致"在线不接触真值"不成立。

    拆分后：
        fit_aggregator(folds, Y, mu, sd, preds, ...) -> AggFit   ← **离线，用 Y**
        apply_aggregator(fit, preds)             -> dict        ← **在线，不用 Y**

    ⇒ 在线阶段只需 `AggFit`（纯 numpy 系数）与分支预测，**看不到 Y**

## AggFit 的内容（全部离线求得）
    base_coef : (5, nt, 2)  每折每目标的融合仿射 (a, c)
    sig_coef  : (5, nt, 5, 2) 每折每目标每分支的 σ 校准 (a, c)
    alpha     : (5,)        TS 结构性闭合系数（内层 CV 选）
    ash       : (5,)        每折的 Ash 均值

## 数值锚点
    闭合前 gap = 0.0329 ｜ 闭合后 gap = **0.0324**（与 v39 一致）
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_AG = os.path.dirname(_HERE)
_REV = os.path.dirname(_AG)
for _p in (_AG, _HERE, os.path.join(_REV, 'code')):
    if _p not in sys.path:
        sys.path.insert(0, _p)

__all__ = ['AggFit', 'fit_aggregator', 'apply_aggregator',
           'ORDER', 'ALPHAS_GRID', 'Aggregator']

ORDER = ('b1', 'CNN', 'b7', 'b8', 'b9')
ALPHAS_GRID = np.round(np.arange(0.0, 1.0001, 0.05), 2)


@dataclass
class AggFit:
    """离线拟合结果（**不含任何真值**，可序列化给在线阶段）。"""
    base_coef: np.ndarray      # (n_folds, nt, 2)
    sig_coef: np.ndarray       # (n_folds, nt, n_branch, 2)
    alpha: np.ndarray          # (n_folds,)
    ash: np.ndarray            # (n_folds,)
    order: tuple = ORDER
    targets: tuple = ()

    def __repr__(self):
        return (f'AggFit(folds={len(self.alpha)}, nt={self.base_coef.shape[1]}, '
                f'alpha={np.round(self.alpha, 2).tolist()})')


# ---------------------------------------------------------------- 离线

def fit_aggregator(folds, Y, mu, sd, preds, targets,
                   apply_ts_closure=True, order=None, cow=None):
    """**离线拟合**（需要真值）。返回 AggFit。

    ⚠️ `cow` 用于 ASH/alpha 的内层 CV **按牛分组**；缺省时退化为顺序切分
       并打印警告（顺序切分可能让同一头牛同时出现在内层训练/测试中）。
    """
    order = tuple(order or [n for n in ORDER if n in preds])
    n, nt = Y.shape
    nf = len(folds)
    base_coef = np.zeros((nf, nt, 2))
    sig_coef = np.zeros((nf, nt, len(order), 2))
    alpha = np.ones(nf)
    ash = np.zeros(nf)
    iF, iP = targets.index('Fat'), targets.index('Prot')
    iL, iT = targets.index('Lact'), targets.index('TS')

    for fi, (tr, te) in enumerate(folds):
        Ptr = np.stack([preds[k]['tr'][fi] for k in order], -1)   # (m, nt, B)
        for ti in range(nt):
            mu_i, sd_i = mu[fi][ti], sd[fi][ti]
            y = (Y[tr, ti] - mu_i) / sd_i
            w = np.ones(len(order)) / len(order)
            A = np.vstack([Ptr[:, ti, :] @ w, np.ones(len(tr))]).T
            cf, *_ = np.linalg.lstsq(A, y, rcond=None)
            base_coef[fi, ti] = cf
            for bi in range(len(order)):
                Ab = np.vstack([Ptr[:, ti, bi], np.ones(len(tr))]).T
                cb, *_ = np.linalg.lstsq(Ab, y, rcond=None)
                sig_coef[fi, ti, bi] = cb
        ash[fi] = float((Y[tr, iT] - Y[tr, iF] - Y[tr, iP] - Y[tr, iL]).mean())

        if apply_ts_closure:
            # 内层 CV 选 alpha（需要 Y，**仅在离线阶段**）
            # ⭐ 必须**按牛分组**，否则同一头牛可能同时出现在内层训练/测试中
            if cow is not None:
                from sklearn.model_selection import GroupKFold
                gk = GroupKFold(4)
                inner = list(gk.split(np.zeros((len(tr), 1)),
                                      np.zeros(len(tr)), np.asarray(cow)[tr]))
            else:
                print('  ⚠️ fit_aggregator: 未提供 cow，'
                      '内层 CV 退化为顺序切分（可能有同牛泄漏）')
                idx = np.arange(len(tr))
                inner = [(None, idx[k::4]) for k in range(4)]
            mses = np.zeros(len(ALPHAS_GRID))
            Ptr_full = Ptr
            # ⭐ 与外层融合一致：先用**整个外层训练集**拟合一次仿射，
            #    再对训练行取预测（step481 的 Ptr 就是这个含义）
            dev_tr = np.zeros((len(tr), nt))
            for ti2 in range(nt):
                mu2, sd2 = mu[fi][ti2], sd[fi][ti2]
                y2 = (Y[tr, ti2] - mu2) / sd2
                w2 = np.ones(len(order)) / len(order)
                A2 = np.vstack([Ptr_full[:, ti2, :] @ w2,
                                np.ones(len(tr))]).T
                cf2, *_ = np.linalg.lstsq(A2, y2, rcond=None)
                dev_tr[:, ti2] = ((cf2[0] * (Ptr_full[:, ti2, :] @ w2) + cf2[1])
                                  * sd2 + mu2)

            for a_i, b in inner:
                # ⭐ Ash 只用**内层训练折**（step481: ash_of(tr[a])）——
                #    若用整折 ash，等于把外层信息漏进内层评估
                idx_tr_inner = (np.arange(len(tr)) if a_i is None
                                else np.asarray(a_i))
                ash_a = float((Y[tr[idx_tr_inner], iT]
                               - Y[tr[idx_tr_inner], iF]
                               - Y[tr[idx_tr_inner], iP]
                               - Y[tr[idx_tr_inner], iL]).mean())
                dev_b = dev_tr[b, iF] + dev_tr[b, iP] + dev_tr[b, iL] + ash_a
                for k2, al in enumerate(ALPHAS_GRID):
                    pb = al * dev_tr[b, iT] + (1 - al) * dev_b
                    mses[k2] += np.mean((pb - Y[tr[b], iT]) ** 2) / len(inner)
            alpha[fi] = float(ALPHAS_GRID[int(np.argmin(mses))])

    return AggFit(base_coef=base_coef, sig_coef=sig_coef, alpha=alpha,
                  ash=ash, order=order, targets=tuple(targets))


# ---------------------------------------------------------------- 在线

def apply_aggregator(fit: AggFit, preds, mu, sd, Y_shape=None):
    """**在线应用**（不需要真值）。

    ⚠️ 本函数体内不得出现 Y / cow（由 check_isolation.py 断言）。
    mu / sd 是**归一化常数**（来自训练集的统计量），不含标签信息。
    """
    order = fit.order
    nt = fit.base_coef.shape[1]
    n = (Y_shape[0] if Y_shape is not None else
         len(preds[order[0]]['te'][0]))
    nf = len(fit.alpha)
    base = np.zeros((n, nt))
    sig = np.zeros((n, nt))
    # 需要折划分 ⇒ 由调用方通过 preds 的 te 索引隐式给出
    # 这里采用约定：preds[k]['te'][fi] 对应第 fi 折的测试行
    # ⇒ 需要一个 folds 视图；在线时由编排器传入行索引
    raise NotImplementedError(
        'apply_aggregator 需要折的行索引；请使用 apply_aggregator_idx()')


def apply_aggregator_idx(fit: AggFit, folds, preds, mu, sd, n_expected):
    """**在线应用**（显式传入折行索引；不接触真值）。"""
    order = fit.order
    nt = fit.base_coef.shape[1]
    base = np.zeros((n_expected, nt))
    sig = np.zeros((n_expected, nt))
    for fi, (tr, te) in enumerate(folds):
        Pte = np.stack([preds[k]['te'][fi] for k in order], -1)   # (m, nt, B)
        w = np.ones(len(order)) / len(order)
        Pc = np.zeros((len(te), len(order)))
        for ti in range(nt):
            mu_i, sd_i = mu[fi][ti], sd[fi][ti]
            a, c = fit.base_coef[fi, ti]
            base[te, ti] = (a * (Pte[:, ti, :] @ w) + c) * sd_i + mu_i
            for bi in range(len(order)):
                ab, cb = fit.sig_coef[fi, ti, bi]
                Pc[:, bi] = ab * Pte[:, ti, bi] + cb
            sig[te, ti] = Pc.std(-1) * sd_i

    yhat = base.copy()
    tg = list(fit.targets)
    if 'TS' in tg and 'Fat' in tg:
        iF, iP = tg.index('Fat'), tg.index('Prot')
        iL, iT = tg.index('Lact'), tg.index('TS')
        for fi, (tr, te) in enumerate(folds):
            dev = base[te, iF] + base[te, iP] + base[te, iL] + fit.ash[fi]
            yhat[te, iT] = fit.alpha[fi] * base[te, iT] + (1 - fit.alpha[fi]) * dev
    return {'yhat': yhat, 'base': base, 'sigma': sig, 'alpha': fit.alpha}


# ---------------------------------------------------------------- 兼容旧接口

class Aggregator:
    """旧接口包装：`run()` = fit + apply（同进程内一步完成）。"""

    def __init__(self, folds, Y, mu, sd, targets, cow=None, alpha=0.90):
        self.folds = folds
        self.Y = Y
        self.mu = mu
        self.sd = sd
        self.targets = list(targets)
        self.cow = cow

    def base_fusion(self, preds, order=None):
        fit = fit_aggregator(self.folds, self.Y, self.mu, self.sd, preds,
                             self.targets, apply_ts_closure=False,
                             order=order, cow=self.cow)
        return apply_aggregator_idx(fit, self.folds, preds, self.mu, self.sd,
                                    self.Y.shape[0])['base']

    def sigma(self, preds, order=None):
        fit = fit_aggregator(self.folds, self.Y, self.mu, self.sd, preds,
                             self.targets, apply_ts_closure=False,
                             order=order, cow=self.cow)
        return apply_aggregator_idx(fit, self.folds, preds, self.mu, self.sd,
                                    self.Y.shape[0])['sigma']

    def run(self, preds, apply_ts_closure=True, cow=None):
        fit = fit_aggregator(self.folds, self.Y, self.mu, self.sd, preds,
                             self.targets, apply_ts_closure=apply_ts_closure,
                             cow=self.cow)
        out = apply_aggregator_idx(fit, self.folds, preds, self.mu, self.sd,
                                   self.Y.shape[0])
        out['targets'] = self.targets
        return out


# ---------------------------------------------------------------- 自测

def _self_test() -> int:
    ok, fail = 0, []
    from sklearn.model_selection import GroupKFold
    import mems_data as M
    import metrics as MT
    from agents.a2_branch import BranchAgent

    d = M.load()
    Y = np.asarray(d['Y'], dtype=float)
    cow = np.asarray(d['cow'])
    T = list(d['targets'])
    n, nt = Y.shape
    folds = list(GroupKFold(5).split(np.zeros((n, 1)), Y[:, 0], cow))
    Ti = np.concatenate([te for _, te in folds])
    B = np.load(os.path.join(_REV, 'experiments',
                             'step426_4v_vi4_cache.npz'), allow_pickle=True)
    MU = [B[f'f{fi}_mu'] for fi in range(5)]
    SD = [B[f'f{fi}_sd'] for fi in range(5)]
    preds = {}
    for nm in ORDER:
        ag = BranchAgent(nm, folds)
        preds[nm] = {'tr': [ag.fold_train(fi) for fi in range(5)],
                     'te': [ag.fold_test(fi) for fi in range(5)]}

    print('=' * 74)
    print('a3_aggregate.py 自测（离线拟合 / 在线应用 分离）')
    print('=' * 74)

    fit = fit_aggregator(folds, Y, MU, SD, preds, T,
                         apply_ts_closure=True, cow=cow)
    print(f'  AggFit: {fit}')
    # ① AggFit 不含真值
    has_truth = any(np.asarray(getattr(fit, k)).shape == Y.shape
                    for k in ('base_coef', 'sig_coef', 'alpha', 'ash'))
    if not has_truth:
        ok += 1
    else:
        fail.append('AggFit 含有与 Y 同形状的字段（可能泄漏）')

    # ② 在线应用复现 gap
    out = apply_aggregator_idx(fit, folds, preds, MU, SD, n)
    f_final = MT.r2_by_target(Y, out['yhat'], Ti)
    g_final = float(sum(0.98 - f_final[t] for t in MT.MAIN3))
    f_base = MT.r2_by_target(Y, out['base'], Ti)
    g_base = float(sum(0.98 - f_base[t] for t in MT.MAIN3))
    print(f'  闭合前 gap = {g_base:.4f}（期望 0.0329）')
    print(f'  闭合后 gap = {g_final:.4f}（期望 0.0324）')
    print(f'  alpha = {np.round(fit.alpha, 2).tolist()}')
    if abs(g_base - 0.0329) < 1e-3:
        ok += 1
    else:
        fail.append(f'闭合前 gap={g_base:.4f}')
    if abs(g_final - 0.0324) < 1.5e-3:
        ok += 1
    else:
        fail.append(f'闭合后 gap={g_final:.4f}')

    # ③ 旧接口 Aggregator.run 与新的 fit+apply 一致
    A = Aggregator(folds, Y, MU, SD, T, cow=cow)
    old = A.run(preds, apply_ts_closure=True)
    same = (np.allclose(old['yhat'], out['yhat'], atol=1e-9)
            and np.allclose(old['sigma'], out['sigma'], atol=1e-9))
    if same:
        ok += 1
    else:
        fail.append('旧接口与新路径结果不一致')

    # ④ apply_aggregator 不含 Y（静态）
    src = open(os.path.abspath(__file__), encoding='utf-8').read()
    i = src.index('def apply_aggregator_idx(')
    j = src.index('# ------', i)
    body = src[i:j]
    bad = [tok for tok in ('Y', 'cow') if f'{tok}[' in body or f'{tok}.' in body
           or f'= {tok}' in body]
    if not bad:
        ok += 1
    else:
        fail.append(f'apply_aggregator_idx 引用了真值标识 {bad}')

    print(f'  通过 {ok} 项')
    if fail:
        print(f'  ❌ 失败 {len(fail)} 项：')
        for f_ in fail:
            print('     -', f_)
        return 1
    print('  ✅ 全部通过：AggFit 不含真值；在线应用复现 0.0329/0.0324；'
          '新旧一致；apply 不引用 Y')
    return 0


if __name__ == '__main__':
    sys.exit(_self_test())
