# -*- coding: utf-8 -*-
"""context.py — 逐 Agent 的输入白名单（上下文隔离）

## 为什么需要
    AGENT_CHARTER.md 规则 A1：**真值只在校验器中可见**。
    若评估/路由 Agent 能看到真值，第四章的"可靠性评估"就不再是盲的，
    全部结论作废。

## 设计
    · 每个 Agent 有**唯一允许的数据键集合**（白名单）
    · `Context.for_agent(name)` 返回一个**只读守卫视图**
    · 访问白名单外的键 ⇒ 抛 `ContextViolation`（不是 KeyError，语义更明确）
    · 写入 ⇒ 抛 `ContextReadOnly`

## 用法
    ctx = Context.from_store({'Y': Y, 'yhat': PRED, 'sigma': SIG, ...})
    a = ctx.for_agent('assess')      # a['yhat'], a['sigma'] 可读；a['Y'] 抛错
    v = ctx.for_agent('validator')   # v['Y'] 可读

## 自测
    python context.py     # 跑内置判据检查
"""
from __future__ import annotations

import os
import sys

__all__ = ['Context', 'GuardedView', 'ContextViolation', 'ContextReadOnly',
           'AGENT_ACCESS', 'AGENT_ORDER']

# ---------------------------------------------------------------- 异常

class ContextViolation(PermissionError):
    """访问了该 Agent 无权访问的数据键。"""


class ContextReadOnly(PermissionError):
    """尝试写入只读的 Context 视图。"""


# ---------------------------------------------------------------- 白名单

#: 每个 Agent 允许看到的数据键（AGENT_CHARTER.md 规则 A1）
AGENT_ACCESS = {
    # 采集诊断：只看原始谱与波长
    'acquisition': frozenset({'X_raw', 'wl', 'view_names'}),

    # 分支推理：只看与自己有关的视图与该分支模型
    'branch': frozenset({'X_views', 'wl', 'view_names', 'branch_name',
                         'branch_model', 'mu', 'sd'}),

    # 汇总：只看各分支输出
    'aggregate': frozenset({'branch_preds', 'mu', 'sd', 'branch_names',
                            'train_index', 'test_index', 'Y_train'}),

    # 评估：只看点预测与 sigma（**不含真值**）
    'assess': frozenset({'yhat', 'sigma', 'target_names', 'mu', 'sd',
                         'train_index', 'test_index', 'Y_train'}),

    # 路由：只看区间、容许限、阈值（**不含真值**）
    'route': frozenset({'intervals_lo', 'intervals_hi', 'yhat', 'tolerances',
                        'decision_thresholds', 'target_names', 'cost_params',
                        'test_index', 'Y_train'}),

    # 报告：只看路由决策（**不含真值**）
    'report': frozenset({'decisions', 'intervals_lo', 'intervals_hi', 'yhat',
                         'target_names', 'degraded', 'missing_branches'}),

    # 校验：**唯一可见真值**
    'validator': frozenset({'report', 'Y', 'cow', 'yhat', 'intervals_lo',
                            'intervals_hi', 'decisions', 'target_names'}),

    # 回流：标记高价值样本（参考值来自实验室，**不用 Y**）
    # ⭐ 回流只需 validator 确认过的**样本 ID**，不需要真值本身（规则 A1）
    'feedback': frozenset({'decisions', 'confirmed_sample_ids', 'sample_ids'}),
}

#: 执行顺序（供 orchestrator 参考）
AGENT_ORDER = ('acquisition', 'branch', 'aggregate', 'assess', 'route',
               'report', 'validator', 'feedback')

#: 绝对禁止除 validator 外任何 Agent 访问的键（硬约束，防止白名单写错）
TRUTH_KEYS = frozenset({'Y', 'cow', 'reference_values'})
TRUTH_PRIVILEGED = frozenset({'validator'})


# ---------------------------------------------------------------- 守卫视图

class GuardedView(dict):
    """只含白名单键的只读视图；越权访问抛 ContextViolation。"""

    __slots__ = ('_agent', '_allowed')

    def __init__(self, data: dict, allowed: frozenset, agent: str):
        self._agent = agent
        self._allowed = allowed
        # 只把允许的键放进去（其余根本不暴露）
        super().__init__({k: v for k, v in data.items() if k in allowed})

    # -- 读 --
    def __getitem__(self, key):
        if key not in self._allowed:
            raise ContextViolation(
                f'agent {self._agent!r} 无权访问 {key!r}；'
                f'允许的键：{sorted(self._allowed)}')
        return super().__getitem__(key)

    def get(self, key, default=None):
        if key not in self._allowed:
            raise ContextViolation(
                f'agent {self._agent!r} 无权访问 {key!r}（get 同样受限）')
        return super().get(key, default)

    def __contains__(self, key):
        # 越权探测也视为违规，避免"用 in 绕过"
        if key not in self._allowed:
            raise ContextViolation(
                f'agent {self._agent!r} 无权探测 {key!r}')
        return super().__contains__(key)

    # -- 写 --
    def __setitem__(self, key, value):
        raise ContextReadOnly(f'agent {self._agent!r} 的视图是只读的')

    def __delitem__(self, key):
        raise ContextReadOnly(f'agent {self._agent!r} 的视图是只读的')

    def update(self, *a, **k):
        raise ContextReadOnly(f'agent {self._agent!r} 的视图是只读的')

    def pop(self, *a, **k):
        raise ContextReadOnly(f'agent {self._agent!r} 的视图是只读的')

    def __repr__(self):
        return f'<GuardedView agent={self._agent!r} keys={sorted(self)}>'


# ---------------------------------------------------------------- Context

class Context:
    """数据仓 + 逐 Agent 白名单视图工厂。"""

    def __init__(self, store: dict):
        self._store = dict(store)
        self._issued = []

    # -- 构造 --
    @classmethod
    def from_store(cls, store: dict) -> 'Context':
        return cls(store)

    @classmethod
    def from_npz(cls, paths, extra=None) -> 'Context':
        """从一批 .npz 载入（后载入者覆盖先载入者）。"""
        import numpy as np
        store = {}
        if isinstance(paths, (str, bytes, os.PathLike)):
            paths = [paths]
        for p in paths:
            with np.load(p, allow_pickle=True) as z:
                for k in z.files:
                    store[k] = z[k]
        if extra:
            store.update(extra)
        return cls(store)

    # -- 发券 --
    def for_agent(self, agent: str) -> GuardedView:
        if agent not in AGENT_ACCESS:
            raise KeyError(f'未知 agent：{agent!r}；已知：{sorted(AGENT_ACCESS)}')
        allowed = AGENT_ACCESS[agent]
        # 硬约束：真值键只允许 validator
        if agent not in TRUTH_PRIVILEGED:
            leak = allowed & TRUTH_KEYS
            assert not leak, (f'白名单配置错误：{agent} 被允许访问真值键 {leak}；'
                              f'见 AGENT_CHARTER 规则 A1')
        self._issued.append(agent)
        return GuardedView(self._store, allowed, agent)

    # -- 内省 --
    @property
    def available(self):
        return sorted(self._store)

    @property
    def issued(self):
        return list(self._issued)


# ---------------------------------------------------------------- 自测

def _self_test():
    import numpy as np
    ok = 0
    fail = []

    store = {
        'Y': np.zeros((5, 3)), 'cow': np.array([1, 1, 2, 2, 3]),
        'yhat': np.ones((5, 3)), 'sigma': np.full((5, 3), 0.1),
        'intervals_lo': np.zeros((5, 3)), 'intervals_hi': np.ones((5, 3)),
        'tolerances': {'Fat': 0.3}, 'decision_thresholds': {'Fat': [3.5]},
        'target_names': ['Fat', 'Prot', 'TS'], 'mu': np.zeros(3), 'sd': np.ones(3),
        'branch_preds': np.zeros((5, 3, 5)), 'branch_names': ['b1'] * 5,
        'train_index': np.array([0, 1]), 'test_index': np.array([2, 3, 4]),
        'Y_train': np.zeros((2, 3)), 'decisions': [], 'cost_params': {},
        'report': {}, 'reference_values': {}, 'sample_ids': [],
        'confirmed_sample_ids': [], 'degraded': False,
        'missing_branches': [], 'view_names': ['v'], 'X_raw': {}, 'X_views': {},
        'wl': {}, 'branch_name': 'b1', 'branch_model': None,
    }
    ctx = Context.from_store(store)

    # 判据 1：assess 取 Y 必须抛 ContextViolation
    a = ctx.for_agent('assess')
    try:
        _ = a['Y']
        fail.append('assess 能读 Y（应抛错）')
    except ContextViolation:
        ok += 1

    # 判据 2：route 取 Y 必须抛错
    r = ctx.for_agent('route')
    try:
        _ = r['Y']
        fail.append('route 能读 Y（应抛错）')
    except ContextViolation:
        ok += 1

    # 判据 3：report 取 Y 必须抛错
    rp = ctx.for_agent('report')
    try:
        _ = rp['Y']
        fail.append('report 能读 Y（应抛错）')
    except ContextViolation:
        ok += 1

    # 判据 4：validator 可取 Y
    v = ctx.for_agent('validator')
    try:
        _ = v['Y']
        ok += 1
    except ContextViolation:
        fail.append('validator 不能读 Y（应允许）')

    # 判据 5：正常键可读
    try:
        assert a['yhat'].shape == (5, 3)
        assert a['sigma'].shape == (5, 3)
        ok += 1
    except Exception as e:
        fail.append(f'assess 读 yhat/sigma 失败：{e}')

    # 判据 6：只读
    try:
        a['yhat'] = None
        fail.append('视图可写入（应只读）')
    except ContextReadOnly:
        ok += 1

    # 判据 7：`in` 探测也受限
    try:
        _ = 'Y' in a
        fail.append("用 'Y' in assess 未被拦截")
    except ContextViolation:
        ok += 1

    # 判据 8：白名单自检——除 validator 外无人可及真值
    try:
        for name, allowed in AGENT_ACCESS.items():
            if name in TRUTH_PRIVILEGED:
                continue
            leak = allowed & TRUTH_KEYS
            assert not leak, f'{name} 泄漏 {leak}'
        ok += 1
    except AssertionError as e:
        fail.append(f'白名单配置错误：{e}')

    print('=' * 72)
    print('context.py 自测')
    print('=' * 72)
    print(f'  通过 {ok} 项')
    if fail:
        print(f'  ❌ 失败 {len(fail)} 项：')
        for f_ in fail:
            print('     -', f_)
        return 1
    print('  ✅ 全部通过：真值仅 validator 可见；越权访问被拦截；视图只读')
    return 0


if __name__ == '__main__':
    sys.exit(_self_test())
