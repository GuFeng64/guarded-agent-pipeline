# -*- coding: utf-8 -*-
"""mems_data.py — **合成数据加载器**（公开仓库版）

## ⚠️ 这是替身，不是原始数据
    原始研究数据为第三方乳品光谱数据集，不可再分发。
    本文件生成**结构完全相同的合成数据**，用于让代码可导入、可演示。

## 接口契约（与原始加载器一致）
    load() -> dict
        'X'       : {view_name: ndarray (n, L)}
        'Y'       : ndarray (n, 5)      5 个目标
        'cow'     : ndarray (n,)        分组键（每头牛多条记录）
        'targets' : list[str]
        'wl'      : {view_name: ndarray (L,)}  波长

## 合成数据的构造原则
    · **光谱与目标之间必须有真实可学的关系**，否则演示毫无意义
      ⇒ Y 由 3 个潜在因子生成；每条光谱 = 潜在因子的线性投影 + 噪声 + 基线
      ⇒ 这样任何回归器都能学到结构
    · 各视图的长度与原始一致（126 / 201 / 201 / 226）
    · 目标范围与乳品成分的常识一致
"""
from __future__ import annotations

import numpy as np

__all__ = ['load', 'describe', 'VIEWS', 'TARGETS']

VIEWS = ('Multi SE14', 'Multi_new SE20', 'Reflection SE20', 'Multi SE25')
LENGTHS = {'Multi SE14': 126, 'Multi_new SE20': 201,
           'Reflection SE20': 201, 'Multi SE25': 226}
RANGES = {'Multi SE14': (1100, 1350), 'Multi_new SE20': (1550, 1950),
          'Reflection SE20': (1550, 1950), 'Multi SE25': (2000, 2450)}
TARGETS = ['Fat', 'Prot', 'Lact', 'TS', 'Urea']
#: 各目标的典型均值与标准差（乳品成分常识量级）
TARGET_MEAN = np.array([4.00, 3.45, 4.85, 12.60, 22.0])
TARGET_SD = np.array([0.95, 0.42, 0.16, 1.10, 6.5])

_N = 297
_N_COW = 71
_SEED = 20260101


def _rng(seed=_SEED):
    return np.random.default_rng(seed)


def _make_targets(rng):
    """由 3 个潜在因子生成 5 个目标。

    ⭐ 用潜在因子而非独立随机，保证**目标之间存在相关**（真实的乳品成分相关），
       也让光谱能同时解释多个目标。
    """
    z = rng.normal(size=(_N, 3))
    # 目标=潜在因子的线性组合（列是目标，行是因子）
    W = np.array([
        [0.90, 0.10, -0.20, 0.85, 0.10],   # 因子1：乳固体
        [-0.20, 0.85, 0.30, 0.30, 0.20],   # 因子2：蛋白
        [0.10, 0.20, -0.70, 0.10, 0.85],   # 因子3：乳糖/尿素
    ])
    Z = z @ W
    Z = (Z - Z.mean(0)) / (Z.std(0) + 1e-12)
    Y = TARGET_MEAN[None, :] + TARGET_SD[None, :] * Z
    # 物理约束：TS ≈ Fat + Prot + Lact + Ash
    iF, iP, iL, iT = 0, 1, 2, 3
    ash = 1.05
    Y[:, iT] = Y[:, iF] + Y[:, iP] + Y[:, iL] + ash
    # Urea 独立一点（避免完美可解）
    # Urea 与 Lact 加入**独立的强噪声**，使其难以从光谱预测。
    # ⭐ 这复现了真实数据的关键特征：Lact/Urea 的 RPD 只有 1.59/1.07
    #    （不可报告），而 Fat/Prot/TS 的 RPD 达 5–6（可报告）。
    #    TS 之所以仍然好预测：其方差主要由 Fat/Prot 主导（Lact 的标准差只有 0.16）。
    Y[:, 2] = Y[:, 2] + rng.normal(0, 0.16 * 2.2, _N)     # Lact：噪声 > 信号
    # Urea：**完全独立于光谱**（只用独立噪声）⇒ 复现真实的 RPD ≈ 1.07
    # ⚠️ 若仍由潜在因子生成，光谱就能看见它，Urea 会变得可预测（已实测踩坑）
    Y[:, 4] = TARGET_MEAN[4] + rng.normal(0, TARGET_SD[4], _N)
    return Y, z


def _make_cow(rng):
    """每头牛 4–5 条记录"""
    cow = np.repeat(np.arange(_N_COW), int(np.ceil(_N / _N_COW)))[:_N]
    rng.shuffle(cow)
    return cow


def _make_spectrum(rng, wl, z, Y):
    """一个视图的光谱：潜在因子 + 目标信号 的线性投影 + 平滑噪声 + 基线"""
    L = len(wl)
    xn = (wl - wl.min()) / (wl.max() - wl.min())
    # 平滑的载荷函数（模拟吸收峰）
    basis = np.stack([
        np.exp(-((xn - 0.30) ** 2) / 0.02),
        np.exp(-((xn - 0.55) ** 2) / 0.03),
        np.exp(-((xn - 0.75) ** 2) / 0.025),
        np.sin(2 * np.pi * xn) * 0.5,
        np.cos(3 * np.pi * xn) * 0.3,
    ], axis=1)                      # (L, 5)
    # 潜在因子（3 个）的载荷：取前 3 个基函数
    load_lat = basis[:, :3] @ np.array([[1.0, 0.3, 0.0],
                                        [0.2, 1.0, 0.3],
                                        [0.0, 0.4, 1.0]])
    sig = z @ load_lat.T            # (n, L) 来自潜在因子
    # 目标相关的附加信号：**只让 Fat/Prot/TS 进入光谱**
    # ⭐ Urea 与 Lact 的独立噪声**不得**投影进光谱，否则它们就变得可预测了
    #    （已实测踩坑：Urea 本是纯噪声，却因被投影而拿到 R²=0.82）
    mask = np.array([1.0, 1.0, 0.0, 1.0, 0.0])
    sig += 0.25 * (((Y - TARGET_MEAN) / TARGET_SD) * mask) @ basis.T
    # 平滑噪声
    noise = rng.normal(0, 0.15, size=(_N, L))
    k = np.hanning(9)
    k = k / k.sum()
    for i in range(_N):
        noise[i] = np.convolve(noise[i], k, mode='same')
    # 基线漂移（SNV 会消除大部分，但保留真实感）
    baseline = 0.3 * xn[None, :] * rng.normal(0, 1, (_N, 1))
    return (sig + noise + baseline).astype(np.float64)


def load(seed=_SEED):
    """返回与原始加载器一致的字典。"""
    rng = _rng(seed)
    Y, z = _make_targets(rng)
    cow = _make_cow(rng)
    X, wl = {}, {}
    for v in VIEWS:
        a, b = RANGES[v]
        w = np.linspace(a, b, LENGTHS[v])
        wl[v] = w
        X[v] = _make_spectrum(rng, w, z, Y)
    return {'X': X, 'Y': Y, 'cow': cow, 'targets': list(TARGETS), 'wl': wl}


def describe():
    d = load()
    out = []
    out.append(f'样本数 {d["Y"].shape[0]} ｜ 目标 {d["targets"]}')
    out.append(f'分组数（牛） {len(np.unique(d["cow"]))}')
    for v in VIEWS:
        out.append(f'  {v:18} X {d["X"][v].shape}  波长 '
                   f'{d["wl"][v][0]:.0f}–{d["wl"][v][-1]:.0f} nm')
    out.append('目标范围：')
    for i, t in enumerate(TARGETS):
        y = d['Y'][:, i]
        out.append(f'  {t:6} {y.min():8.3f} – {y.max():8.3f}  '
                   f'均值 {y.mean():8.3f}  标准差 {y.std():8.3f}')
    return '\n'.join(out)


if __name__ == '__main__':
    print('=' * 70)
    print('SYNTHETIC DATA（结构同原始，数值不可用于科研结论）')
    print('=' * 70)
    print(describe())
    # 自检：可学性（**模拟真实流程**：SNV + 岭回归，而非原始最小二乘）
    d = load()
    Y = d['Y']
    X0 = d['X']['Multi_new SE20']
    # SNV（与真实管线一致）
    Xs = (X0 - X0.mean(1, keepdims=True)) / (X0.std(1, keepdims=True) + 1e-9)
    tr = np.arange(0, 220)
    te = np.arange(220, 297)
    # 岭回归（201 维 + 220 样本，必须正则化）
    Xtr, Xte = Xs[tr], Xs[te]
    lam = 10.0
    A = Xtr.T @ Xtr + lam * np.eye(Xtr.shape[1])
    b = Xtr.T @ (Y[tr] - Y[tr].mean(0))
    coef = np.linalg.solve(A, b)
    pred = Xte @ coef + Y[tr].mean(0)
    print()
    print('  可学性自检（SNV + 岭回归，held-out）：')
    for i, t in enumerate(TARGETS):
        r2 = 1 - np.sum((Y[te, i] - pred[:, i]) ** 2) / np.sum(
            (Y[te, i] - Y[te, i].mean()) ** 2)
        print(f'    {t:6} R² = {r2:+.3f}')
    print('  （Fat/Prot/TS 的 R² 明显 > 0 ⇒ 合成数据确有可学结构）')
