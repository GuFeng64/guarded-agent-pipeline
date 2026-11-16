# -*- coding: utf-8 -*-
"""agents/a4_assess.py — 评估 Agent：保形区间 + 按 RPD 分级

## 职责（确定性；**不接触真值以外的任何标签推测**）
    输入：yhat（含 TS 闭合的点预测）
    输出：
        lo, hi  (n, nt)   保形区间（默认 90% / 95%）
        q       (nt,)     每目标的保形分位数
        grades  dict      逐目标的"可报告性"分级（来自 RPD）
        coverage dict     名义 vs 实测覆盖率

## 关键设计
    ① **常数宽度**保形区间（本项目实测：σ 归一化反而更宽 16–23%）
    ② cross-conformal：校准集 = **其余折**的 OOF 残差
       有限样本修正：k = ceil((m+1)(1−α))
    ③ RPD 分级门槛（Williams）：>3 可报告 / 2.5–3 良好 / 2–2.5 筛查 / <2 不可用
       ⭐ 分级决定该目标能否进入报告，与 a5_route 的决策相互独立

## 数值锚点（自测断言）
    名义 90% ⇒ 实测覆盖率 90.6%（±1.5%）

## 自测
    python a4_assess.py
"""
from __future__ import annotations

import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_AG = os.path.dirname(_HERE)
_REV = os.path.dirname(_AG)
for _p in (_AG, _HERE, os.path.join(_REV, 'code')):
    if _p not in sys.path:
        sys.path.insert(0, _p)

__all__ = ['Assessor', 'RPD_GRADES', 'grade_of_rpd']

#: RPD 分级（Williams–Norris 惯例）
RPD_GRADES = (
    (3.0, 'report', '可报告（质量控制级）'),
    (2.5, 'good', '良好（可报告，需注明）'),
    (2.0, 'screening', '仅筛查（不得用于结算）'),
    (0.0, 'reject', '不可用（不报数值，转参考法）'),
)


def grade_of_rpd(rpd: float):
    for thr, code, desc in RPD_GRADES:
        if rpd > thr:
            return code, desc
    return 'reject', '不可用'


class Assessor:
    """评估 Agent 的实现（确定性，无 LLM）。"""

    def __init__(self, folds, Y, targets):
        self.folds = folds
        self.Y = Y
        self.targets = list(targets)
        self.ix = {t: i for i, t in enumerate(self.targets)}
        self.n, self.nt = Y.shape

    # ---------- 保形区间 ----------
    def conformal_quantile(self, yhat, ti, alpha):
        """返回每折的 q（常数宽度），以及用于评估的 (lo, hi)。"""
        lo = np.full(self.n, np.nan)
        hi = np.full(self.n, np.nan)
        qs = []
        for fi, (tr, te) in enumerate(self.folds):
            cal = np.setdiff1d(np.arange(self.n), te)
            s = np.abs(self.Y[cal, ti] - yhat[cal, ti])
            m = len(s)
            k = min(max(int(np.ceil((m + 1) * (1 - alpha))), 1), m)
            q = float(np.sort(s)[k - 1])
            qs.append(q)
            lo[te] = yhat[te, ti] - q
            hi[te] = yhat[te, ti] + q
        return lo, hi, float(np.mean(qs))

    def intervals(self, yhat, alpha=0.10):
        LO = np.full((self.n, self.nt), np.nan)
        HI = np.full((self.n, self.nt), np.nan)
        Q = np.full(self.nt, np.nan)
        for t in self.targets:
            ti = self.ix[t]
            lo, hi, q = self.conformal_quantile(yhat, ti, alpha)
            LO[:, ti], HI[:, ti], Q[ti] = lo, hi, q
        return LO, HI, Q

    # ---------- 覆盖率 ----------
    def coverage(self, yhat, LO, HI):
        Ti = np.concatenate([te for _, te in self.folds])
        out = {}
        for t in self.targets:
            ti = self.ix[t]
            out[t] = float(np.mean((self.Y[Ti, ti] >= LO[Ti, ti])
                                   & (self.Y[Ti, ti] <= HI[Ti, ti])))
        return out

    # ---------- RPD 分级 ----------
    def metrics_and_grades(self, yhat):
        Ti = np.concatenate([te for _, te in self.folds])
        r2, rmse, rpd, grades = {}, {}, {}, {}
        for t in self.targets:
            ti = self.ix[t]
            y, p = self.Y[Ti, ti], yhat[Ti, ti]
            e = p - y
            rmse[t] = float(np.sqrt(np.mean(e ** 2)))
            r2[t] = float(1.0 - np.sum(e ** 2)
                          / (np.sum((y - y.mean()) ** 2) + 1e-12))
            rpd[t] = float(y.std() / (rmse[t] + 1e-12))
            code, desc = grade_of_rpd(rpd[t])
            grades[t] = {'r2': r2[t], 'rmse': rmse[t], 'rpd': rpd[t],
                         'code': code, 'desc': desc,
                         'reportable': code in ('report', 'good')}
        return r2, rmse, rpd, grades

    # ---------- 一体化 ----------
    def run(self, yhat, alpha=0.10):
        LO, HI, Q = self.intervals(yhat, alpha)
        cov = self.coverage(yhat, LO, HI)
        r2, rmse, rpd, grades = self.metrics_and_grades(yhat)
        return {'lo': LO, 'hi': HI, 'q': Q, 'coverage': cov,
                'r2': r2, 'rmse': rmse, 'rpd': rpd, 'grades': grades,
                'alpha': alpha}


# ---------------------------------------------------------------- 自测

def _self_test() -> int:
    ok, fail = 0, []
    from sklearn.model_selection import GroupKFold
    import mems_data as M

    d = M.load()
    Y = np.asarray(d['Y'], dtype=float)
    cow = np.asarray(d['cow'])
    T = list(d['targets'])
    n, nt = Y.shape
    folds = list(GroupKFold(5).split(np.zeros((n, 1)), Y[:, 0], cow))

    # 用 v39 的最终预测（含 TS 闭合）
    Z = np.load(os.path.join(_REV, 'experiments', 'step481_ts_alpha.npz'),
                allow_pickle=True)
    yhat = Z['pred_blend']

    A = Assessor(folds, Y, T)
    print('=' * 72)
    print('a4_assess.py 自测')
    print('=' * 72)

    out = A.run(yhat, alpha=0.10)

    # ① 名义 90% ⇒ 实测覆盖率 ≈ 90.6%（主判据三目标均值）
    cov3 = float(np.mean([out['coverage'][t] for t in ('Fat', 'Prot', 'TS')]))
    print(f'  名义 90% ⇒ 三主目标实测覆盖率 {cov3:.1%}（期望 90.6%）')
    if abs(cov3 - 0.906) < 0.015:
        ok += 1
    else:
        fail.append(f'覆盖率 {cov3:.1%}，期望 90.6%±1.5%')

    # ② 95% 档
    out95 = A.run(yhat, alpha=0.05)
    cov3_95 = float(np.mean([out95['coverage'][t] for t in ('Fat', 'Prot', 'TS')]))
    print(f'  名义 95% ⇒ 三主目标实测覆盖率 {cov3_95:.1%}（期望 95.2%±1.5%）')
    if abs(cov3_95 - 0.952) < 0.015:
        ok += 1
    else:
        fail.append(f'95% 覆盖率 {cov3_95:.1%}')

    # ③ 区间必须包含点预测
    inside = all(np.all(out['lo'][:, i] <= yhat[:, i] + 1e-9)
                 and np.all(yhat[:, i] <= out['hi'][:, i] + 1e-9)
                 for i in range(nt))
    if inside:
        ok += 1
    else:
        fail.append('区间未包含点预测')

    # ④ 宽度在**折内**为常数（cross-conformal 的 q 是逐折估计的，
    #    因此折间宽度可以不同；这是设计如此，不是缺陷）
    wid = out['hi'] - out['lo']
    const_in_fold = True
    for fi, (tr, te) in enumerate(folds):
        for i in range(nt):
            if np.ptp(wid[te, i]) > 1e-9:
                const_in_fold = False
    if const_in_fold:
        ok += 1
    else:
        fail.append('区间宽度在折内不是常数（与设计不符）')
    print(f'  折内宽度范围（Fat）: '
          f'{[round(float(np.ptp(wid[te, 0])), 10) for _, te in folds]}')
    print(f'  折间 q 变化（Fat）: '
          f'{[round(float(wid[te, 0][0]), 4) for _, te in folds]}')

    # ⑤ RPD 分级与 v39 的已知值一致
    print('  RPD 分级：')
    for t in T:
        g = out['grades'][t]
        print(f'    {t:5} R²={g["r2"]:+.4f} RPD={g["rpd"]:5.2f} '
              f'→ {g["code"]:9} {g["desc"]}')
    exp = {'Fat': 'report', 'Prot': 'report', 'TS': 'report',
           'Lact': 'reject', 'Urea': 'reject'}
    ok5 = all(out['grades'][t]['code'] == exp[t] for t in T)
    if ok5:
        ok += 1
    else:
        fail.append('RPD 分级与预期不符：'
                    + str({t: out['grades'][t]['code'] for t in T}))

    print(f'  通过 {ok} 项')
    if fail:
        print(f'  ❌ 失败 {len(fail)} 项：')
        for f_ in fail:
            print('     -', f_)
        return 1
    print('  ✅ 全部通过：90% 覆盖率达标；区间常数宽度且含点预测；RPD 分级正确')
    return 0


if __name__ == '__main__':
    sys.exit(_self_test())
