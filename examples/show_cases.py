# -*- coding: utf-8 -*-
"""examples/show_cases.py — 五个门控案例的完整追踪

## 目的
    一页看懂"门控真的在拦东西"。每个案例打印：
        提议（原文）→ 哪条规则触发 → 拒绝原因 → 最终决策

## 用法
    python examples/show_cases.py

## 案例
    C1  G1  非法动作          提议 ship_it
    C2  G2  不可报告目标       对 Lact 提议 report
    C3  G5  编造数值          提议里出现工具从未返回的数
    C4  G6  report 是最差动作  期望损失显示该复测却选了报告
    C5  G7  批量预算击穿       10 个样本全 retest，预算 20%
    C6  审议环                 提议违规 → 审计质疑 → 修订 → 通过
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
_AS = os.path.join(_REPO, 'agent_system')
for _p in (_AS, os.path.join(_AS, 'agents'), os.path.join(_REPO, 'code')):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from agents.a5_guarded import Guard, GuardedAgent, Proposal          # noqa: E402
from agents.a5_deliberation import CriticAgent                       # noqa: E402

W = 76


def hr(title):
    print()
    print('─' * W)
    print(title)
    print('─' * W)


def show_proposal(p, indent='  '):
    print(f'{indent}提议动作：{json.dumps(p.actions, ensure_ascii=False)}')
    if p.reasons:
        print(f'{indent}理由码　：{json.dumps(p.reasons, ensure_ascii=False)}')
    if p.raw:
        txt = p.raw if len(p.raw) <= 120 else p.raw[:117] + '...'
        print(f'{indent}提议原文：{txt}')


def show_verdict(g, indent='  '):
    print(f'{indent}门控结果：{"通过" if g.ok else "拒绝"}')
    if g.violations:
        for v in g.violations:
            print(f'{indent}  ▸ {v}')
    failed = [k for k, v in g.checked.items() if not v]
    if failed:
        print(f'{indent}失败规则：{failed}')


def mk_ctx():
    """一份真实结构的上下文（合成数值）"""
    T = ['Fat', 'Prot', 'Lact', 'TS', 'Urea']
    return {
        'targets': T,
        'lo': np.array([3.9512, 3.4051, 4.8012, 12.2043, 20.10]),
        'hi': np.array([4.3507, 3.6033, 5.0021, 12.6051, 30.02]),
        'sigma': np.array([0.1004, 0.0502, 0.0503, 0.1002, 2.501]),
        'q': np.array([0.3120, 0.1130, 0.1600, 0.3540, 8.2520]),
        'grades': {
            'Fat': {'code': 'report', 'reportable': True, 'rpd': 5.25, 'r2': 0.9638},
            'Prot': {'code': 'report', 'reportable': True, 'rpd': 6.13, 'r2': 0.9734},
            'Lact': {'code': 'reject', 'reportable': False, 'rpd': 1.59, 'r2': 0.6044},
            'TS': {'code': 'report', 'reportable': True, 'rpd': 5.81, 'r2': 0.9704},
            'Urea': {'code': 'reject', 'reportable': False, 'rpd': 1.07, 'r2': 0.1268},
        },
        'grade_thr': {'Fat': (4.0, 4.5, 5.0), 'Prot': (3.4, 3.7, 4.0),
                      'TS': (13.5, 14.5, 15.5), 'Lact': (), 'Urea': ()},
        'compliance': {'Fat': 3.1, 'Prot': 2.9, 'TS': None,
                       'Lact': None, 'Urea': None},
        'regret': {'Fat': {'report': 1.0, 'retest': 2.0, 'reference': 5.0},
                   'Prot': {'report': 0.8, 'retest': 1.5, 'reference': 4.0},
                   'TS': {'report': 1.2, 'retest': 2.2, 'reference': 6.0}},
        'degraded': False, 'missing': [],
    }


OK_ACTIONS = {'Fat': 'report', 'Prot': 'report', 'Lact': 'reference',
              'TS': 'report', 'Urea': 'reference'}
OK_REASONS = {'Fat': ['NONE'], 'Prot': ['NONE'], 'Lact': ['LOW_RPD'],
              'TS': ['NONE'], 'Urea': ['LOW_RPD']}


def main():
    print('=' * W)
    print('门控案例追踪 —— 五个"提议被拦下"的真实轨迹')
    print('=' * W)

    ctx = mk_ctx()
    G = Guard()

    # ---------- 基准：合法提议 ----------
    hr('基准  一个合法提议（用于对照）')
    base = Proposal(actions=dict(OK_ACTIONS), reasons=dict(OK_REASONS),
                    source='llm', n_tool_calls=18)
    show_proposal(base)
    show_verdict(G.check(base, ctx))

    # ---------- C1 ----------
    hr('C1  G1 非法动作 —— 提议 "ship_it"')
    p1 = Proposal(actions={**OK_ACTIONS, 'Fat': 'ship_it'},
                  reasons=dict(OK_REASONS), source='llm')
    show_proposal(p1)
    show_verdict(G.check(p1, ctx))
    print('  → 最终决策：回退规则（source=rule）')

    # ---------- C2 ----------
    hr('C2  G2 不可报告目标 —— 对 Lact 提议 report')
    p2 = Proposal(actions={**OK_ACTIONS, 'Lact': 'report'},
                  reasons=dict(OK_REASONS), source='llm')
    show_proposal(p2)
    show_verdict(G.check(p2, ctx))
    print('  → 最终决策：回退规则；Lact 的 RPD=1.59 ⇒ 只能 reference')

    # ---------- C3 ----------
    hr('C3  G5 编造数值 —— 提议里出现工具从未返回的数')
    p3 = Proposal(actions=dict(OK_ACTIONS), reasons=dict(OK_REASONS),
                  source='llm',
                  raw='Fat 区间宽度 0.3995，σ=0.1004，预计误差约 0.8765，可报告')
    show_proposal(p3)
    show_verdict(G.check(p3, ctx))
    print('  → 0.3995 与 0.1004 可溯源（区间与 σ 由工具返回），')
    print('     0.8765 从未被任何工具返回 ⇒ G5 拒绝')

    # ---------- C4 ----------
    hr('C4  G6 report 是最差动作 —— 期望损失显示该复测')
    ctx4 = mk_ctx()
    ctx4['regret']['Fat'] = {'report': 9.0, 'retest': 1.0, 'reference': 2.0}
    p4 = Proposal(actions=dict(OK_ACTIONS), reasons=dict(OK_REASONS),
                  source='llm')
    show_proposal(p4)
    print('  期望损失（Fat）：report 9.00 ｜ retest 1.00 ｜ reference 2.00')
    show_verdict(G.check(p4, ctx4))
    print('  → 最终决策：回退规则 → Fat 改为 retest')

    # ---------- C5 ----------
    hr('C5  G7 批量预算击穿 —— 10 个样本全提 retest，预算 20%')
    ctxs = [mk_ctx() for _ in range(10)]

    class _AllRetest(GuardedAgent):
        def decide(self, ctx):
            p = Proposal(
                actions={t: ('retest' if t in ('Fat', 'Prot', 'TS')
                             else 'reference') for t in ctx['targets']},
                reasons={t: ['HIGH_SIGMA'] for t in ctx['targets']},
                source='llm', n_tool_calls=6)
            return p, Guard().check(p, ctx), {'llm_used': True, 'guard_ok': True}

    props, diags = _AllRetest().decide_batch(ctxs, budget_frac=0.20, verbose=True)
    n_ret = sum(1 for p in props if any(a == 'retest' for a in p.actions.values()))
    n_rule = sum(1 for p in props if p.source == 'rule')
    print(f'  每条提议单独看都合法（G1–G6 全过）')
    print(f'  整批：10/10 提 retest，预算 = 2')
    print(f'  → 最终：retest {n_ret} 个 ｜ 回退规则 {n_rule} 个')
    print(f'  → G7 按期望损失降序保留预算内的，其余回退')

    # ---------- C6 ----------
    hr('C6  审议环 —— 提议违规 → 审计质疑 → 修订 → 通过')
    print('  【第 1 轮】提议方给出：Lact=report（违规）')
    p6a = Proposal(actions={'Fat': 'report', 'Prot': 'report', 'Lact': 'report',
                            'TS': 'report', 'Urea': 'reference'},
                   reasons={t: ['NONE'] for t in ctx['targets']}, source='llm')
    show_proposal(p6a, indent='    ')
    crit = CriticAgent().review(p6a, ctx, rnd=0)
    print(f'  【审计 Agent】ok={crit.ok}')
    for h in crit.hints:
        print(f'    质疑：{h}')
    print()
    print('  【第 2 轮】提议方收到质疑后修订：')
    p6b = Proposal(actions=dict(OK_ACTIONS), reasons=dict(OK_REASONS),
                   source='llm')
    show_proposal(p6b, indent='    ')
    g6 = G.check(p6b, ctx)
    print(f'  【审计 Agent】ok={g6.ok}  ⇒ 收敛，采纳')

    # ---------- 汇总 ----------
    print()
    print('=' * W)
    print('六个案例：4 个被逐样本门控拦下，1 个被批量门控裁剪，1 个经修订后通过')
    print('=' * W)


if __name__ == '__main__':
    main()
