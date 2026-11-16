"""metrics.py — 主判据的【唯一实现】。

## 为什么需要这个文件（连续三次同类错误）

    ① R² 算在全量 297 行上而不是测试行 → 得到 −88 / −724 的假数字
    ② 残差均值漂移的诊断：用了含 Lact 的聚合
    ③ step299：主判据写成 KEEP（含 Lact）而非 MAIN3
       → 打印出 gap 0.4595 的假象，差点把真提升（0.0386）当成失败

    **教训：主判据只能有一份实现。所有脚本调用它，不许各自重算。**

## 章程定义（§00.1i A0.4）

    主判据 = (0.98 − R²_Fat) + (0.98 − R²_Prot) + (0.98 − R²_TS)
    **不含 Lact**（Lact 在 MEMS 上数据封顶，见 §00.1k）
    **不含 Urea**（信噪比极低）

    ⚠️ 也**不要**用"四目标均值"——它被 Lact 等权重稀释，会把人引向无用的优化。
"""
import numpy as np
from sklearn.metrics import r2_score

TARGETS = ['Fat', 'Prot', 'Lact', 'TS', 'Urea']
MAIN3 = ('Fat', 'Prot', 'TS')
TERTIARY = ('Lact', 'Urea')
TARGET_LEVEL = 0.98


def r2_by_target(Y, Pred, rows, targets=TARGETS):
    """R² per target, computed ONLY on `rows` (the tested rows)."""
    return {t: float(r2_score(Y[rows, TARGETS.index(t)], Pred[rows, TARGETS.index(t)]))
            for t in targets}


def main_criterion(Y, Pred, rows):
    """the ONE main criterion: sum over Fat/Prot/TS of (0.98 − R²).

    Lower is better. Requires `rows` explicitly so that a caller cannot accidentally
    score on all rows.
    """
    rows = np.asarray(rows)
    r = r2_by_target(Y, Pred, rows, MAIN3)
    gap = sum(TARGET_LEVEL - r[t] for t in MAIN3)
    return {'r2': r, 'gap': float(gap)}


def report(name, Y, Pred, rows):
    """one-line summary in the canonical format used across all step logs."""
    c = main_criterion(Y, Pred, rows)
    r = c['r2']
    return (f'{name:34} Fat {r["Fat"]:+.4f}  Prot {r["Prot"]:+.4f}  TS {r["TS"]:+.4f}  '
            f'gap {c["gap"]:.4f}')


def assert_correct_usage(rows, n):
    """guard: `rows` must be the tested rows, never the full row set by accident."""
    rows = np.asarray(rows)
    if rows.size == n:
        # allowed only when every row really is tested; make the caller say so explicitly
        raise ValueError(
            'metrics called with all rows. If every row truly is tested, pass '
            'np.arange(n) explicitly after verifying the OOF construction.')
    return True
