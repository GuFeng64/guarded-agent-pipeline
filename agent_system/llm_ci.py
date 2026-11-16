# -*- coding: utf-8 -*-
"""llm_ci.py — LLM 推理层指标的**置信区间**（不改动实验，只改报告方式）

## 为什么需要
    文档现在写"阈值跨越准确率 55.6%"、"CFS 3.4%"。
    但样本量极小（n=18 次判断、29 次消融）⇒ 点估计**不能单独报告**：
      55.6% = 10/18 的 95% Wilson 区间约 [33%, 76%]
      CFS 3.4% = 1/29 的 95% 精确二项区间约 [0.1%, 17%]
    ⇒ 正确写法是"点估计（n, 95% CI）"。

## 本模块
    · `wilson_ci(k, n)`       —— 比例的 Wilson 区间（小样本比正态近似更稳）
    · `exact_binom_ci(k, n)`  —— Clopper–Pearson 精确区间
    · `report(k, n)`          —— 一行可粘贴的报告字符串

## 用法
    python llm_ci.py
"""
from __future__ import annotations

import math
import sys

__all__ = ['wilson_ci', 'exact_binom_ci', 'report']


def _z(conf):
    # 双侧正态分位数（常用值，避免依赖 scipy）
    table = {0.90: 1.6448536269514722,
             0.95: 1.959963984540054,
             0.99: 2.5758293035489004}
    return table.get(round(conf, 2), 1.959963984540054)


def wilson_ci(k, n, conf=0.95):
    """比例的 Wilson 得分区间（推荐用于小样本）。"""
    if n <= 0:
        return (float('nan'), float('nan'))
    z = _z(conf)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def _betacf(a, b, x, itmax=200, eps=3e-16):
    """连分式（Numerical Recipes），用于不完全 Beta 函数"""
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    if abs(d) < 1e-300:
        d = 1e-300
    d = 1.0 / d
    h = d
    for m in range(1, itmax + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < 1e-300:
            d = 1e-300
        c = 1.0 + aa / c
        if abs(c) < 1e-300:
            c = 1e-300
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < 1e-300:
            d = 1e-300
        c = 1.0 + aa / c
        if abs(c) < 1e-300:
            c = 1e-300
        d = 1.0 / d
        de = d * c
        h *= de
        if abs(de - 1.0) < eps:
            break
    return h


def _betainc(a, b, x):
    """正则化不完全 Beta 函数 I_x(a,b)"""
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    lbeta = (math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
             + a * math.log(x) + b * math.log(1 - x))
    if x < (a + 1) / (a + b + 2):
        return math.exp(lbeta) * _betacf(a, b, x) / a
    return 1.0 - math.exp(lbeta) * _betacf(b, a, 1 - x) / b


def _beta_ppf(q, a, b, lo=0.0, hi=1.0, it=200):
    """Beta 分布分位数（二分法；足够精确且无 scipy 依赖）"""
    for _ in range(it):
        mid = (lo + hi) / 2
        if _betainc(a, b, mid) < q:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def exact_binom_ci(k, n, conf=0.95):
    """Clopper–Pearson 精确区间（保守，适合极小样本）。"""
    if n <= 0:
        return (float('nan'), float('nan'))
    alpha = 1 - conf
    lo = 0.0 if k == 0 else _beta_ppf(alpha / 2, k, n - k + 1)
    hi = 1.0 if k == n else _beta_ppf(1 - alpha / 2, k + 1, n - k)
    return (lo, hi)


def report(k, n, label='', conf=0.95, method='wilson'):
    """生成'点估计 (n=…, 95% CI …)'格式的字符串"""
    if n <= 0:
        return f'{label}: n/a'
    p = k / n * 100
    lo, hi = (wilson_ci(k, n, conf) if method == 'wilson'
              else exact_binom_ci(k, n, conf))
    s = (f'{label}{p:.1f}% (n={n}, {int(conf * 100)}% '
         f'CI {lo * 100:.1f}–{hi * 100:.1f}%)')
    return s


# ---------------------------------------------------------------- 自测

def _self_test() -> int:
    ok, fail = 0, []
    print('=' * 76)
    print('llm_ci.py 自测')
    print('=' * 76)

    # ① Wilson 对称性/端点
    lo, hi = wilson_ci(0, 20)
    print(f'  0/20  Wilson: [{lo:.4f}, {hi:.4f}]')
    if abs(lo) < 1e-9 and hi < 0.2:
        ok += 1
    else:
        fail.append('Wilson 下端点异常')
    lo2, hi2 = wilson_ci(20, 20)
    print(f'  20/20 Wilson: [{lo2:.4f}, {hi2:.4f}]')
    if abs(hi2 - 1.0) < 1e-9:
        ok += 1
    else:
        fail.append('Wilson 上端点异常')

    # ② 与已知值比对（10/18 的 Wilson 95% 约 [0.337, 0.751]）
    lo3, hi3 = wilson_ci(10, 18)
    print(f'  10/18 Wilson: [{lo3:.4f}, {hi3:.4f}]  （期望约 0.337, 0.751）')
    if abs(lo3 - 0.337) < 0.01 and abs(hi3 - 0.751) < 0.01:
        ok += 1
    else:
        fail.append(f'Wilson 与已知值不符 {lo3:.3f},{hi3:.3f}')

    # ③ Clopper–Pearson：1/29 应很宽
    lo4, hi4 = exact_binom_ci(1, 29)
    print(f'  1/29  CP:    [{lo4:.4f}, {hi4:.4f}]')
    if lo4 < 0.01 and hi4 > 0.10:
        ok += 1
    else:
        fail.append(f'CP 区间异常 {lo4:.4f},{hi4:.4f}')

    # ④ 报告字符串
    s = report(10, 18, '准确率 ')
    print(f'  报告示例：{s}')
    if 'n=18' in s and 'CI' in s:
        ok += 1
    else:
        fail.append('报告字符串格式异常')

    # ⑤ 大样本应更窄
    w_small = wilson_ci(10, 18)
    w_big = wilson_ci(100, 180)
    width_s = w_small[1] - w_small[0]
    width_b = w_big[1] - w_big[0]
    print(f'  区间宽度：n=18 → {width_s:.3f} ｜ n=180 → {width_b:.3f}')
    if width_b < width_s / 2.5:
        ok += 1
    else:
        fail.append('区间宽度未随 n 收缩')

    print(f'  通过 {ok} 项')
    if fail:
        print(f'  ❌ 失败 {len(fail)} 项：')
        for f_ in fail:
            print('     -', f_)
        return 1
    print('  ✅ 全部通过：Wilson/精确区间计算正确，n 增大时区间收窄')
    return 0


def _report_existing():
    """报告现有 6 样本实验的区间（数据来自 logs/agent_llm_eval2.txt）"""
    print('=' * 76)
    print('现有 LLM 推理层结果的置信区间（样本量极小，必须报告区间）')
    print('=' * 76)
    items = [
        ('阈值跨越准确率', 10, 18, 'wilson'),
        ('建议与规则一致率', 12, 18, 'wilson'),
        ('ESI 重复一致性', 16, 18, 'wilson'),
        ('CFS 概念翻转率', 1, 29, 'exact'),
        ('PSS 改写一致性', 15, 18, 'wilson'),
        ('契约遵守', 18, 18, 'wilson'),
    ]
    for lab, k, n, meth in items:
        print('  ' + report(k, n, f'{lab:16} ', method=meth))
    print()
    print('  ⚠️ 结论：除"契约遵守"外，所有指标的 95% CI 宽度都在 ±20pp 以上')
    print('     ⇒ **不得以点估计形式单独写入论文**；扩样需 API 密钥。')


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == '--report':
        _report_existing()
        sys.exit(0)
    sys.exit(_self_test())
