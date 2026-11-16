# -*- coding: utf-8 -*-
"""agents/a1_acquisition.py — 采集 Agent：基于实测的检查项 + 尖峰修复

## 版本历史（重要：避免重复踩坑）
    v1：检查散射/基线/饱和 ⇒ **无效**
        step515/516 实测：SNV 对乘性散射与加性偏移**数学上完全免疫**
        （snv(a·x)=snv(x)，snv(x+b)=snv(x)，SNV 后与干净谱距离均为 0.0000）
    v2：检查项改为
        ① 视图完整性  ② 非有限值  ③ 饱和  **④ 尖峰/毛刺**（唯一实测有效）
        尖峰 ⇒ 逐点定点替换修复；修复后可接受 ⇒ ACCEPT(corrected=True)
        修复不足 ⇒ RESHOOT

## 尖峰指标（经实测校准）
    spike_index = max|x − smooth| / std(x)      ← **相对谱自身尺度**

    ⚠️ 不要用 max(residual)/robust_sd(residual)：那是**自归一化**的，
       尖峰越少比值反而越大（3 个尖峰比 8 个更"严重"），跨情形不可比。

| 情形 | spike_index |
|---|---|
| 干净谱 | median 0.0009–0.029 ｜ p99 0.062 ｜ **跨视图 max 0.073** |
| 注入尖峰（未修） | 2.3 – 2.6 |
| 修复后 | 0.28 – 0.50 |

    ⇒ 双阈值：soft=0.10 检出（干净 0/297 假阳性）；hard=1.0 判修复后可否接受

## 修复（step516 实测回收率 50.8%）
    定点替换 + 迭代：每轮重算平滑参考，只替换被标记的点，保留真实谱峰

## 自测
    python a1_acquisition.py
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np
from scipy.signal import medfilt, savgol_filter

_HERE = os.path.dirname(os.path.abspath(__file__))
_AG = os.path.dirname(_HERE)
_REV = os.path.dirname(_AG)
for _p in (_AG, _HERE, os.path.join(_REV, 'code')):
    if _p not in sys.path:
        sys.path.insert(0, _p)

__all__ = ['AcquisitionAgent', 'AcqResult', 'DEFAULT_LIMITS',
           'spike_index', 'repair_spikes']

RAW_VIEWS = ('Multi SE14', 'Multi_new SE20', 'Reflection SE20', 'Multi SE25')

#: 阈值（**基于实测校准**，见模块头注释）
DEFAULT_LIMITS = {
    'saturation_frac': 0.995,
    'spike_soft': 0.10,      # 检出阈值（实测：干净谱跨视图 max 0.073）
    'spike_hard': 1.0,       # 修复后仍超过 ⇒ RESHOOT
    'median_width': 5,
    'repair_iter': 6,
    'point_thresh': 0.5,     # 逐点判定阈值（相对 std）
}


def _smooth_ref(x, width=5):
    """参考平滑谱（中值 + Savitzky–Golay 平均）"""
    x = np.asarray(x, dtype=float)
    w = width if width % 2 == 1 else width + 1
    w = max(3, min(w, (len(x) // 2) * 2 - 1))
    med = medfilt(x, w)
    wl = min(11, (len(x) // 2) * 2 - 1)
    sg = savgol_filter(x, wl, 2) if wl >= 5 else med
    return 0.5 * (med + sg)


def spike_index(x, width=5):
    """尖峰指标：max|x − smooth| / std(x)（相对谱自身尺度，跨情形可比）"""
    x = np.asarray(x, dtype=float)
    if x.size < 9:
        return 0.0
    r = np.abs(x - _smooth_ref(x, width))
    return float(np.max(r) / (np.std(x) + 1e-12))


def repair_spikes(x, point_thresh=0.5, width=5, n_iter=6):
    """定点替换 + 迭代（每轮重算平滑参考，处理相邻尖峰）"""
    x = np.asarray(x, dtype=float).copy()
    total = 0
    for _ in range(n_iter):
        sd = np.std(x) + 1e-12
        r = np.abs(x - _smooth_ref(x, width)) / sd
        idx = np.where(r > point_thresh)[0]
        if idx.size == 0:
            break
        x[idx] = _smooth_ref(x, width)[idx]
        total += int(idx.size)
    return x, total


@dataclass
class AcqResult:
    verdict: str                       # 'ACCEPT' | 'RESHOOT'
    reasons: List[str] = field(default_factory=list)
    checks: Dict[str, bool] = field(default_factory=dict)
    corrected: bool = False
    spike_index: float = float('nan')
    repaired: Dict[str, np.ndarray] = field(default_factory=dict)

    def __repr__(self):
        tag = ' (corrected)' if self.corrected else ''
        return f'AcqResult({self.verdict}{tag}, reasons={self.reasons})'


class AcquisitionAgent:
    """采集 Agent（确定性规则）。"""

    def __init__(self, limits=None, try_repair=True):
        self.lim = dict(DEFAULT_LIMITS)
        if limits:
            self.lim.update(limits)
        self.try_repair = try_repair

    def check_one(self, X_views: dict) -> AcqResult:
        res = AcqResult(verdict='ACCEPT')
        w = self.lim['median_width']

        # ① 视图完整性
        missing = [v for v in RAW_VIEWS if v not in X_views]
        res.checks['complete'] = not missing
        if missing:
            res.reasons.append(f'缺少视图 {missing}')

        # ② 非有限值
        has_nan = any(not np.all(np.isfinite(np.asarray(x, dtype=float)))
                      for x in X_views.values())
        res.checks['finite'] = not has_nan
        if has_nan:
            res.reasons.append('存在非有限值')

        # ③ 饱和
        sat = [v for v, x in X_views.items()
               if np.asarray(x).size
               and np.nanmax(np.asarray(x, dtype=float)) >= self.lim['saturation_frac']]
        res.checks['no_saturation'] = not sat
        if sat:
            res.reasons.append(f'{sat}: 可能饱和')

        # ④ 尖峰
        worst, worst_v = -1.0, None
        for v, x in X_views.items():
            if not np.all(np.isfinite(np.asarray(x, dtype=float))):
                continue
            r = spike_index(x, w)
            if r > worst:
                worst, worst_v = r, v
        res.spike_index = worst
        detected = worst > self.lim['spike_soft']
        res.checks['no_spikes'] = not detected
        if detected:
            res.reasons.append(f'{worst_v}: 尖峰（index={worst:.3f}）')

        # 尝试修复
        if detected and self.try_repair:
            fixed, nfix = {}, 0
            for v, x in X_views.items():
                xr, nf = repair_spikes(x, self.lim['point_thresh'], w,
                                       self.lim['repair_iter'])
                fixed[v] = xr
                nfix += nf
            r2 = max(spike_index(x, w) for x in fixed.values())
            res.spike_index = r2
            if r2 <= self.lim['spike_hard']:
                res.repaired = fixed
                res.corrected = True
                res.checks['no_spikes'] = True
                res.reasons.append(
                    f'已定点替换 {nfix} 点（index {worst:.3f}→{r2:.3f}）')
            else:
                res.reasons.append(f'修复不足（index 仍 {r2:.3f}）')

        res.verdict = 'ACCEPT' if all(res.checks.values()) else 'RESHOOT'
        return res


# ---------------------------------------------------------------- 自测

def _self_test() -> int:
    ok, fail = 0, []
    import mems_data as M

    d = M.load()
    X = d['X']
    n = X[RAW_VIEWS[0]].shape[0]
    A = AcquisitionAgent()

    print('=' * 72)
    print('a1_acquisition.py 自测（基于实测的检查项）')
    print('=' * 72)

    v = {k: X[k][0] for k in RAW_VIEWS}

    # ① 正常 ⇒ ACCEPT，未修正
    r0 = A.check_one(v)
    print(f'  正常样本: {r0.verdict}  spike_index={r0.spike_index:.4f}  '
          f'corrected={r0.corrected}')
    if r0.verdict == 'ACCEPT' and not r0.corrected:
        ok += 1
    else:
        fail.append(f'正常样本被判 {r0.verdict}')

    # ② 缺视图
    v1 = dict(v); v1.pop('Reflection SE20')
    r1 = A.check_one(v1)
    if r1.verdict == 'RESHOOT' and any('缺少视图' in x for x in r1.reasons):
        ok += 1
    else:
        fail.append('缺视图未拦下')

    # ③ 饱和
    v2 = dict(v); v2['Multi SE14'] = np.full_like(v['Multi SE14'], 1.0)
    r2 = A.check_one(v2)
    if r2.verdict == 'RESHOOT' and any('饱和' in x for x in r2.reasons):
        ok += 1
    else:
        fail.append('饱和未拦下')

    # ④ NaN
    v3 = dict(v)
    x3 = v['Multi_new SE20'].copy(); x3[10] = np.nan
    v3['Multi_new SE20'] = x3
    if A.check_one(v3).verdict == 'RESHOOT':
        ok += 1
    else:
        fail.append('NaN 未拦下')

    # ⑤ 尖峰 ⇒ 检出 + 修复 + ACCEPT
    v4 = dict(v)
    x4 = v['Multi_new SE20'].copy()
    rng = np.random.default_rng(0)
    idx = rng.choice(len(x4), size=5, replace=False)
    x4[idx] += 3.0 * x4.std()
    v4['Multi_new SE20'] = x4
    r4 = A.check_one(v4)
    print(f'  注入 5 尖峰: {r4.verdict}  corrected={r4.corrected}  '
          f'index={r4.spike_index:.3f}')
    if r4.verdict == 'ACCEPT' and r4.corrected:
        ok += 1
    else:
        fail.append(f'尖峰未被修复（{r4.verdict}, corrected={r4.corrected}）')

    # ⑥ 关闭修复 ⇒ RESHOOT
    A2 = AcquisitionAgent(try_repair=False)
    r5 = A2.check_one(v4)
    if r5.verdict == 'RESHOOT' and any('尖峰' in x for x in r5.reasons):
        ok += 1
    else:
        fail.append(f'关闭修复后尖峰未导致 RESHOOT（{r5.verdict}）')

    # ⑦ 整批：干净样本不应有假阳性
    n_flag = 0
    for i in range(n):
        rr = A.check_one({k: X[k][i] for k in RAW_VIEWS})
        if rr.verdict == 'RESHOOT':
            n_flag += 1
        elif rr.corrected:
            n_flag += 1
    rate = 1 - n_flag / n
    print(f'  整批通过率 {rate:.1%}（{n - n_flag}/{n}，含假阳性/误修正）')
    if rate > 0.95:
        ok += 1
    else:
        fail.append(f'整批通过率过低 {rate:.1%}')

    print(f'  通过 {ok} 项')
    if fail:
        print(f'  ❌ 失败 {len(fail)} 项：')
        for f_ in fail:
            print('     -', f_)
        return 1
    print('  ✅ 全部通过：正常通过；缺视图/饱和/NaN 拦下；尖峰先修复后通过；'
          '关闭修复则拦下')
    return 0


if __name__ == '__main__':
    sys.exit(_self_test())
