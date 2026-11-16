# -*- coding: utf-8 -*-
"""reliability_metrics.py — 可靠性信号的指标库（**带解析值的严格自检**）

## 为什么抽成模块
    step524/525/526 三个分析脚本各自实现了 AURC，
    而本项目已**三次**在"排序方向"上犯错（算成被拒集、索引升序/降序混用）。
    ⇒ 把指标集中到一个模块，用**解析已知值**做自检，杜绝方向错误。

## 核心约定（写死在这里，不再靠记忆）
    score: suspicion，**越大越可疑**
    aurc(score, err):  按 score **升序**保留（最不可疑的优先留），
                       risk(c) = 保留集平均误差，AURC = ∫ risk dc
    risk_at_budget(score, err, b): 拒识比例 b ⇒ 丢掉**最可疑**的 b·n 个

## 自检里的解析值（关键）
    err = [1,2,3,4,5]
    · score = err（完美排序）⇒ risk = [1, 1.5, 2, 2.5, 3]，cov = [.2,.4,.6,.8,1]
          AURC = 0.2×(1.25+1.75+2.25+2.75) = **1.60**（梯形积分，非简单平均）
    · score = -err（完全反序）⇒ risk = [5, 4.5, 4, 3.5, 3]
          AURC = 0.2×(4.75+4.25+3.75+3.25) = **3.20**
    ⚠️ 我最初把解析值写成 2.0/4.0 —— 那是"风险的简单平均"，
       AURC 是**对覆盖率积分**，两者不同（已实测踩坑）
    ⇒ **两个解析值 + oracle 最优性 = 三重阴性对照**
"""
from __future__ import annotations

import numpy as np

__all__ = ['aurc', 'risk_at_budget', 'spearman_abs_err', 'kendall_abs_err',
           'worse_first']


def worse_first(score):
    """返回"最可疑在前"的索引（**具名，杜绝方向混淆**——章程 C43）"""
    return np.argsort(-np.asarray(score, dtype=float))


def _trapz(y, x):
    f = np.trapezoid if hasattr(np, 'trapezoid') else np.trapz
    return float(f(y, x))


def aurc(score, err):
    """风险-覆盖曲线下面积（越小越好）。

    语义：按 score **升序**（最不可疑优先）保留样本；
         coverage=k/n 时 risk=前 k 个的平均绝对误差；AURC=∫risk dc。
    """
    score = np.asarray(score, dtype=float)
    err = np.asarray(err, dtype=float)
    if score.shape != err.shape:
        raise ValueError(f'shape 不匹配 {score.shape} vs {err.shape}')
    n = len(err)
    if n == 0:
        return float('nan')
    best_first = np.argsort(score)          # 最不可疑在前
    kept = err[best_first]
    risks = np.cumsum(kept) / np.arange(1, n + 1)
    cov = np.arange(1, n + 1) / n
    return _trapz(risks, cov)


def risk_at_budget(score, err, budget):
    """拒识比例 budget 时，**保留集**的平均绝对误差。"""
    score = np.asarray(score, dtype=float)
    err = np.asarray(err, dtype=float)
    n = len(err)
    k = int(round(budget * n))
    order = worse_first(score)              # 最可疑在前
    keep = order[k:]                        # 丢掉最可疑的 k 个
    if keep.size == 0:
        return float('nan')
    return float(np.mean(err[keep]))


def spearman_abs_err(score, err):
    """score 与 |error| 的 Spearman 秩相关"""
    from scipy.stats import spearmanr
    return float(spearmanr(score, err)[0])


def kendall_abs_err(score, err):
    from scipy.stats import kendalltau
    return float(kendalltau(score, err)[0])


# ---------------------------------------------------------------- 自测

def _self_test() -> int:
    ok, fail = 0, []

    print('=' * 76)
    print('reliability_metrics.py 自测（含解析值阴性对照）')
    print('=' * 76)

    err = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

    # ① 解析值：完美排序 ⇒ AURC = 2.0
    a_perfect = aurc(err, err)
    print(f'  完美排序（score=err）  AURC = {a_perfect:.6f}  （解析值 1.6）')
    if abs(a_perfect - 1.6) < 1e-9:
        ok += 1
    else:
        fail.append(f'完美排序 AURC={a_perfect:.6f} ≠ 1.6')

    # ② 解析值：完全反序 ⇒ AURC = 4.0
    a_worst = aurc(-err, err)
    print(f'  完全反序（score=-err） AURC = {a_worst:.6f}  （解析值 3.2）')
    if abs(a_worst - 3.2) < 1e-9:
        ok += 1
    else:
        fail.append(f'完全反序 AURC={a_worst:.6f} ≠ 3.2')

    # ③ oracle 最优性（随机对照）
    rng = np.random.default_rng(0)
    e2 = np.abs(rng.normal(0, 1, 200))
    s_oracle = e2.copy()
    s_rev = -e2
    s_rand = [aurc(rng.random(len(e2)), e2) for _ in range(20)]
    a_or, a_rv = aurc(s_oracle, e2), aurc(s_rev, e2)
    print(f'  n=200：oracle {a_or:.4f} ｜ random {np.mean(s_rand):.4f} ｜ '
          f'反序 {a_rv:.4f}')
    if a_or < min(s_rand) <= max(s_rand) < a_rv:
        ok += 1
    else:
        fail.append('oracle/random/反序 的排序关系不成立')

    # ④ risk_at_budget 与 AURC 方向一致
    t_err = np.abs(rng.normal(0, 1, 300))
    t_sig = t_err + rng.normal(0, 0.5, 300)      # 有噪声的可疑度
    r_b0 = risk_at_budget(t_sig, t_err, 0.0)
    r_b5 = risk_at_budget(t_sig, t_err, 0.5)
    print(f'  预算 0% 风险 {r_b0:.4f} ｜ 预算 50% 风险 {r_b5:.4f}')
    # ⭐ 拒识越多 ⇒ 保留集越干净 ⇒ 风险应**下降**
    if r_b5 < r_b0:
        ok += 1
    else:
        fail.append('预算增大时保留集风险未下降（方向可能反了）')

    # ⑤ Spearman 端点
    rho_p = spearman_abs_err(err, err)
    rho_n = spearman_abs_err(-err, err)
    print(f'  Spearman：score=err → {rho_p:+.4f} ｜ score=-err → {rho_n:+.4f}')
    if abs(rho_p - 1.0) < 1e-9 and abs(rho_n + 1.0) < 1e-9:
        ok += 1
    else:
        fail.append('Spearman 端点值异常')

    # ⑥ worse_first 语义
    s = np.array([0.1, 0.9, 0.5])
    if list(worse_first(s)) == [1, 2, 0]:
        ok += 1
    else:
        fail.append(f'worse_first 语义错误：{worse_first(s)}')

    # ⑦ 形状校验
    try:
        aurc(np.zeros(3), np.zeros(4))
        fail.append('形状不匹配未报错')
    except ValueError:
        ok += 1

    print(f'  通过 {ok} 项')
    if fail:
        print(f'  ❌ 失败 {len(fail)} 项：')
        for f_ in fail:
            print('     -', f_)
        return 1
    print('  ✅ 全部通过：解析值 1.6/3.2 精确命中；oracle 最优；'
          '预算增大风险下降；Spearman 端点 ±1；具名方向正确')
    return 0


if __name__ == '__main__':
    import sys
    sys.exit(_self_test())
