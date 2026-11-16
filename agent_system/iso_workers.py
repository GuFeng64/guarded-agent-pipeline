# -*- coding: utf-8 -*-
"""iso_workers.py — 供进程隔离 worker 调用的**纯函数包装**

## 为什么需要
    `process_isolation.IsolatedAgent` 通过 `模块名 + 函数名` 在子进程里构造可调对象。
    `Reporter` / `Assessor` 都是类，需要工厂函数把它们包成"给 kwargs 就能调"的函数。

## 关键约束（隔离的前提）
    本模块**不得在导入时加载任何数据**（不读 npz、不 import mems_data）
    ⇒ 子进程只拿到代码 + 白名单载荷，物理上无真值可访问
"""
from __future__ import annotations

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_AG = os.path.join(_HERE, 'agents')
_REV = os.path.dirname(_HERE)
for _p in (_HERE, _AG, os.path.join(_REV, 'code')):
    if _p not in sys.path:
        sys.path.insert(0, _p)

__all__ = ['render_one', 'grade_one', 'apply_fusion']

# 模块级缓存（子进程内复用，避免每次重建对象）
_REPORTERS = {}


def _reporter(target_names):
    key = tuple(target_names)
    if key not in _REPORTERS:
        from agents.a6_report import Reporter
        _REPORTERS[key] = Reporter(list(target_names))
    return _REPORTERS[key]


def render_one(*, sample_id, yhat_row, lo_row, hi_row, decision, grades,
               target_names, degraded=False, missing=None, alpha=0.10,
               template=None):
    """子进程侧：构造并渲染一份报告文本。"""
    rep = _reporter(target_names)
    r = rep.build(sample_id=sample_id, yhat_row=yhat_row, lo_row=lo_row,
                  hi_row=hi_row, decision=decision, grades=grades,
                  degraded=degraded, missing=list(missing or []),
                  alpha=alpha)
    return rep.render(r, template=template)


def apply_fusion(*, fit, folds, preds, mu, sd, n_expected):
    """子进程侧：**在线应用**融合（只读 AggFit 系数与分支预测，不接触真值）。

    ⭐ `fit` 是纯 numpy 系数的 dataclass；本函数不引用任何真值。
    """
    from agents.a3_aggregate import apply_aggregator_idx
    return apply_aggregator_idx(fit, folds, preds, mu, sd, n_expected)


def grade_one(*, r2, rpd, target):
    """子进程侧：按 RPD 给出分级（纯函数，无数据依赖）。"""
    from agents.a4_assess import grade_of_rpd
    code, desc = grade_of_rpd(float(rpd))
    return {'code': code, 'desc': desc, 'reportable': code in ('report', 'good')}


if __name__ == '__main__':
    # 自检：本模块导入时不应接触任何数据
    print('iso_workers.py 导入完成；模块级 globals 含 Y:', 'Y' in globals())
