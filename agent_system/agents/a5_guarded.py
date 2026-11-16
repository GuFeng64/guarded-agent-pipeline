# -*- coding: utf-8 -*-
"""agents/a5_guarded.py — **受约束的 Agent**（LLM 提议 + Tool Use + 确定性门控）

## 为什么是这个架构
    本项目实测：LLM 单独决策的阈值跨越判断准确率仅 **46.7%**
    （n=75，95% CI 35.8–57.8%；三分类随机 33%）
    ⇒ **不能**让 LLM 直接决定处置动作
    ⇒ 但也不该完全弃用：LLM 能整合"区间宽度 + 阈值距离 + 退化状态"等多种线索，
       这些线索的组合方式很难用规则穷举

    ⇒ **受约束的 Agent**：
        LLM 看结构化上下文，**通过工具主动查询**它需要的量，
        提出决策 + 理由码；
        确定性门控（Guard）逐条校验，
        通过则采纳，不通过则**回退到规则决策**（Router）。

## 架构
    ┌─────────────┐   tool calls    ┌──────────────┐
    │  LLM 提议   │ ──────────────▶ │  工具层(只读) │
    │ (最多 N 轮) │ ◀────────────── │  无真值访问   │
    └──────┬──────┘   tool results  └──────────────┘
           │ Proposal(action, reasons)
           ▼
    ┌─────────────┐   通过   ┌──────────────┐
    │   Guard     │ ───────▶ │ 采纳 LLM 决策 │
    │ 确定性校验  │  不通过  └──────────────┘
    └──────┬──────┘
           └────────▶ 回退到规则决策（Router）

## 工具（**全部只读，且只能返回结构化字段**）
    get_interval(target)     → 区间 lo/hi/宽度/q/σ
    get_grade(target)        → 可报告性分级 code/rpd/r2
    get_thresholds(target)   → 分档阈值与合规限值
    get_regret(target)       → 三个动作的**期望损失**（来自 A5 路由模型）
    get_status()             → 退化状态（缺失分支、Δgap）

## Guard 的规则（每条都可用阴性对照推翻）
    G1 action ∈ {report, retest, reference}
    G2 不可报告目标不得 report
    G3 degraded 且缺整族 ⇒ 不得 report
    G4 reasons ⊆ CONCEPT_CODES
    G5 proposal 中出现的数值必须能在结构化上下文里溯源
    G6 若 get_regret 显示 report 的期望损失最高 ⇒ 不得 report（防止明显错误）
"""
from __future__ import annotations

import json
import os
import re
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_AG = os.path.dirname(_HERE)
_REV = os.path.dirname(_AG)
for _p in (_AG, _HERE, os.path.join(_REV, 'code')):
    if _p not in sys.path:
        sys.path.insert(0, _p)

__all__ = ['ACTION_CODES', 'CONCEPT_CODES', 'Tools', 'Guard',
           'GuardedAgent', 'Proposal', 'GuardResult']

ACTION_CODES = ('report', 'retest', 'reference')
CONCEPT_CODES = ('THRESHOLD_CROSS', 'WIDE_INTERVAL', 'HIGH_SIGMA',
                 'DEGRADED', 'LOW_RPD', 'NONE')

#: 不可报告目标（与 a4 的 RPD 分级一致）
NON_REPORTABLE = ('Lact', 'Urea')


# ---------------------------------------------------------------- 工具层

class Tools:
    """只读工具。**不接触真值**；只返回已结构化的量。"""

    def __init__(self, ctx: dict):
        """ctx 字段：
            targets        : 目标名列表
            lo, hi         : (nt,) 该样本的区间
            sigma          : (nt,) 该样本的 σ
            q              : (nt,) 保形分位数
            grades         : {target: {code, rpd, r2, reportable}}
            grade_thr      : {target: (lo, mid, hi)} 分档阈值
            compliance     : {target: 限值或 None}
            regret         : {target: {action: expected_loss}}
            degraded       : bool
            missing        : list
        """
        self.ctx = ctx

    # ---- 各工具 ----
    def get_interval(self, target):
        ix = self.ctx['targets'].index(target)
        lo, hi = float(self.ctx['lo'][ix]), float(self.ctx['hi'][ix])
        q = float(self.ctx['q'][ix]) if self.ctx.get('q') is not None else None
        s = float(self.ctx['sigma'][ix])
        return {'target': target, 'lo': lo, 'hi': hi,
                'width': hi - lo, 'q': q, 'sigma': s}

    def get_grade(self, target):
        g = self.ctx['grades'].get(target, {})
        return {'target': target,
                'code': g.get('code'), 'reportable': g.get('reportable'),
                'rpd': _safe(g.get('rpd')), 'r2': _safe(g.get('r2'))}

    def get_thresholds(self, target):
        return {'target': target,
                'grade_thresholds': list(self.ctx.get('grade_thr', {}).get(target, [])),
                'compliance_limit': self.ctx.get('compliance', {}).get(target)}

    def get_regret(self, target):
        r = self.ctx.get('regret', {}).get(target, {})
        return {'target': target,
                **{k: _safe(v) for k, v in r.items()}}

    def get_status(self):
        return {'degraded': bool(self.ctx.get('degraded')),
                'missing': list(self.ctx.get('missing', [])),
                'targets': list(self.ctx['targets'])}

    # ---- 工具注册（供 LLM 的 tool schema 与本地执行）----
    def registry(self):
        return {
            'get_interval': (self.get_interval, {'target': 'str'}),
            'get_grade': (self.get_grade, {'target': 'str'}),
            'get_thresholds': (self.get_thresholds, {'target': 'str'}),
            'get_regret': (self.get_regret, {'target': 'str'}),
            'get_status': (self.get_status, {}),
        }

    def call(self, name, args):
        reg = self.registry()
        if name not in reg:
            raise KeyError(f'未知工具 {name!r}；可用 {sorted(reg)}')
        fn, spec = reg[name]
        # 只接受声明过的参数名
        clean = {k: v for k, v in (args or {}).items() if k in spec}
        out = fn(**clean)
        # 校验返回的 target 必须存在
        if 'target' in out and out['target'] is not None:
            if out['target'] not in self.ctx['targets']:
                raise ValueError(f'未知目标 {out["target"]!r}')
        return out

    def schema(self):
        """给 LLM 的 function-calling schema"""
        out = []
        for name, (fn, spec) in self.registry().items():
            props = {k: {'type': 'string',
                         'enum': list(self.ctx['targets'])}
                     for k, v in spec.items() if v == 'str'}
            out.append({'type': 'function', 'function': {
                'name': name,
                'description': {
                    'get_interval': '查询某目标的预测区间（lo/hi/宽度/q/σ）',
                    'get_grade': '查询某目标的可报告性分级',
                    'get_thresholds': '查询某目标的分档阈值与合规限值',
                    'get_regret': '查询某目标三个动作的期望损失',
                    'get_status': '查询系统退化状态',
                }[name],
                'parameters': {'type': 'object',
                               'properties': props,
                               'required': list(props)},
            }})
        return out


def _safe(v):
    try:
        return None if v is None else float(v)
    except Exception:
        return None


# ---------------------------------------------------------------- 提议与门控

@dataclass
class Proposal:
    actions: Dict[str, str] = field(default_factory=dict)   # target → action
    reasons: Dict[str, List[str]] = field(default_factory=dict)
    source: str = 'llm'          # 'llm' | 'rule'
    n_tool_calls: int = 0
    raw: Any = None

    def __repr__(self):
        return (f'Proposal({self.source}, n_tool={self.n_tool_calls}, '
                f'{self.actions})')


@dataclass
class GuardResult:
    ok: bool
    violations: List[str] = field(default_factory=list)
    checked: Dict[str, bool] = field(default_factory=dict)

    def __repr__(self):
        return f'GuardResult(ok={self.ok}, {self.violations})'


class Guard:
    """确定性门控：逐条校验 LLM 提议。**每条规则都可被阴性对照推翻。**"""

    #: 规则名（供测试逐条注入）
    RULES = ('G1_valid_action', 'G1b_cover_targets',
             'G2_no_report_nonreportable', 'G3_no_report_degraded',
             'G4_valid_reason', 'G5_numbers_traceable',
             'G6_not_worst_action')

    def check(self, prop: Proposal, ctx: dict) -> GuardResult:
        res = GuardResult(ok=True)
        targets = list(ctx['targets'])

        # G1 动作合法性
        bad = {t: a for t, a in prop.actions.items() if a not in ACTION_CODES}
        res.checked['G1_valid_action'] = not bad
        if bad:
            res.violations.append(f'G1 非法动作：{bad}')

        # G1b 必须覆盖所有目标
        miss = [t for t in targets if t not in prop.actions]
        res.checked['G1b_cover_targets'] = not miss
        if miss:
            res.violations.append(f'G1b 缺少目标：{miss}')

        # G2 不可报告目标不得 report
        bad2 = [t for t in NON_REPORTABLE
                if prop.actions.get(t) == 'report']
        res.checked['G2_no_report_nonreportable'] = not bad2
        if bad2:
            res.violations.append(f'G2 不可报告目标被要求 report：{bad2}')

        # G3 退化且缺整族 ⇒ 不得 report
        bad3 = []
        if ctx.get('degraded'):
            miss_b = set(ctx.get('missing', []))
            # 整族缺失（三个 TabPFN 全丢）视为不可报告
            if {'b7', 'b8', 'b9'} <= miss_b:
                bad3 = [t for t in targets if prop.actions.get(t) == 'report']
        res.checked['G3_no_report_degraded'] = not bad3
        if bad3:
            res.violations.append(f'G3 退化状态下不得 report：{bad3}')

        # G4 理由码合法
        bad4 = []
        for t, rs in prop.reasons.items():
            for c in (rs or []):
                if c not in CONCEPT_CODES:
                    bad4.append(f'{t}:{c}')
        res.checked['G4_valid_reason'] = not bad4
        if bad4:
            res.violations.append(f'G4 非法理由码：{bad4[:5]}')

        # G5 提议中出现的数值必须可溯源
        bad5 = self._check_numbers(prop, ctx)
        res.checked['G5_numbers_traceable'] = not bad5
        if bad5:
            res.violations.append(f'G5 数值不可溯源：{bad5[:5]}')

        # G6 若某目标 report 的期望损失最高 ⇒ 不得 report
        bad6 = []
        for t, a in prop.actions.items():
            if a != 'report':
                continue
            r = ctx.get('regret', {}).get(t) or {}
            if len(r) >= 2:
                vals = {k: float(v) for k, v in r.items()
                        if v is not None and np.isfinite(float(v))}
                if vals and vals.get('report') is not None:
                    if vals['report'] >= max(vals.values()) - 1e-12:
                        bad6.append(t)
        res.checked['G6_not_worst_action'] = not bad6
        if bad6:
            res.violations.append(f'G6 report 是期望损失最高的动作：{bad6}')

        res.ok = all(res.checked.values())
        return res

    # ---- G5 实现：数值白名单 ----
    @staticmethod
    def _allowed_numbers(ctx):
        """白名单 = **工具能告诉 LLM 的全部量**。

        ⚠️ 曾经漏掉 `width = hi - lo`（`get_interval` 明确返回它），
           ⇒ LLM 正确引用区间宽度反被判"不可溯源"，导致 100% 回退。
        原则：LLM 只能知道工具说过的数；工具说过的都必须允许。
        """
        allow = set()

        def add(v):
            if v is None:
                return
            try:
                f = float(v)
            except Exception:
                return
            if np.isfinite(f):
                for nd in (0, 1, 2, 3, 4):
                    allow.add(f'{f:.{nd}f}')

        for t in ctx['targets']:
            ix = ctx['targets'].index(t)
            lo, hi = float(ctx['lo'][ix]), float(ctx['hi'][ix])
            add(lo); add(hi)
            add(hi - lo)                    # ⭐ width：get_interval 返回它
            add(ctx['sigma'][ix])
            if ctx.get('q') is not None:
                add(ctx['q'][ix])
            g = ctx['grades'].get(t, {})
            add(g.get('rpd')); add(g.get('r2'))
            for v in (ctx.get('grade_thr', {}).get(t) or []):
                add(v)
            add(ctx.get('compliance', {}).get(t))
            for v in (ctx.get('regret', {}).get(t) or {}).values():
                add(v)
        return allow

    def _check_numbers(self, prop, ctx):
        raw = prop.raw
        if raw is None:
            return []
        s = raw if isinstance(raw, str) else json.dumps(raw, ensure_ascii=False)
        allow = self._allowed_numbers(ctx)
        bad = []
        for tok in re.findall(r'-?\d+\.\d+|-?\d+', s):
            t = tok.lstrip('+').rstrip('.')
            if t in allow:
                continue
            try:
                v = float(t)
            except ValueError:
                continue
            # 小整数（0-9）作序数，豁免
            if t == str(int(v)) and abs(v) <= 9:
                continue
            bad.append(tok)
        return bad


# ---------------------------------------------------------------- Agent 主体

class GuardedAgent:
    """LLM 提议 → 门控 → 采纳或回退。"""

    MAX_ROUNDS = 4

    def __init__(self, router=None, model='deepseek-chat',
                 base_url='https://api.deepseek.com', timeout=60.0):
        self.router = router
        self.model = model
        self.base_url = base_url
        self.timeout = timeout

    # ---- API 可用性 ----
    @property
    def available(self):
        return bool(os.environ.get('LLM_API_KEY'))

    # ---- 主入口 ----
    def decide(self, ctx: dict) -> Tuple[Proposal, GuardResult, Dict]:
        """返回 (最终决策, 门控结果, 诊断信息)"""
        info = {'llm_used': False, 'fallback': False, 'guard_ok': None,
                'n_tool_calls': 0, 'error': None}

        # ① 规则决策（永远先算，作为回退与对照）
        rule = self._rule_proposal(ctx)
        info['rule_actions'] = dict(rule.actions)

        if not self.available:
            info['fallback'] = True
            info['reason'] = '无 LLM 密钥 ⇒ 纯规则'
            return rule, GuardResult(ok=True, checked={'no_llm': True}), info

        # ② LLM 提议
        try:
            prop = self._llm_proposal(ctx, info)
            info['llm_used'] = True
        except Exception as e:
            info['error'] = f'{type(e).__name__}: {e}'
            info['fallback'] = True
            return rule, GuardResult(ok=True, checked={'llm_error': True}), info

        # ③ 门控
        g = Guard().check(prop, ctx)
        info['guard_ok'] = g.ok
        info['violations'] = list(g.violations)

        if g.ok:
            info['adopted'] = 'llm'
            return prop, g, info
        info['fallback'] = True
        info['adopted'] = 'rule'
        return rule, g, info

    # ---- 规则决策（回退）----
    def _rule_proposal(self, ctx) -> Proposal:
        acts, reasons = {}, {}
        for i, t in enumerate(ctx['targets']):
            g = ctx['grades'].get(t, {})
            if not g.get('reportable', False):
                acts[t] = 'reference'
                reasons[t] = ['LOW_RPD']
                continue
            r = ctx.get('regret', {}).get(t) or {}
            if r:
                best = min(r, key=lambda k: (r[k] is None, r[k]))
                acts[t] = best if best in ACTION_CODES else 'report'
            else:
                acts[t] = 'report'
            q = ctx.get('q')
            s = float(ctx['sigma'][i])
            cr = ['HIGH_SIGMA'] if (s > 2 * (float(q[i]) if q is not None
                                             and np.isfinite(q[i]) else 1e-9)) \
                else ['NONE']
            reasons[t] = cr
        return Proposal(actions=acts, reasons=reasons, source='rule')

    # ---- 批量决策（含 G7 预算门控）----
    def decide_batch(self, ctxs, budget_frac=0.20, verbose=False):
        """对一批样本做决策，保证**整批 retest 比例不超过预算**。"""
        n = len(ctxs)
        k_budget = int(round(budget_frac * n))
        props, diags = [], []

        for ctx in ctxs:
            p, g, info = self.decide(ctx)
            props.append(p)
            diags.append(info)

        idx_llm_retest = []
        for i, (p, info) in enumerate(zip(props, diags)):
            if p.source != 'llm':
                continue
            if any(a == 'retest' for a in p.actions.values()):
                idx_llm_retest.append(i)

        n_llm_retest = len(idx_llm_retest)
        over = n_llm_retest > k_budget
        if verbose:
            print(f'  [G7] 预算 {k_budget}/{n} ｜ LLM 提议复测 {n_llm_retest} '
                  f'⇒ {"超预算，裁剪" if over else "在预算内"}')

        if over:
            score = []
            for i in idx_llm_retest:
                ctx = ctxs[i]
                tot = 0.0
                for t, a in props[i].actions.items():
                    if a != 'retest':
                        continue
                    r = ctx.get('regret', {}).get(t) or {}
                    rep = r.get('report')
                    tot += float(rep) if rep is not None else 0.0
                score.append((tot, i))
            score.sort(key=lambda x: -x[0])
            keep = {i for _, i in score[:k_budget]}
            for i in idx_llm_retest:
                if i in keep:
                    continue
                rule = self._rule_proposal(ctxs[i])
                diags[i] = {**diags[i], 'g7_budget_rejected': True,
                            'adopted': 'rule'}
                props[i] = rule

        for i, info in enumerate(diags):
            info.setdefault('g7_budget_rejected', False)
            info['final_source'] = props[i].source

        return props, diags

    # ---- LLM 提议（含 tool-use 循环）----
    def _llm_proposal(self, ctx, info) -> Proposal:
        import requests
        key = os.environ['LLM_API_KEY']
        tools = Tools(ctx)
        schema = tools.schema()

        sys_msg = (
            '你是乳品近红外检测的决策助手。对每个目标在 '
            'report / retest / reference 中选择一个处置动作。\n'
            '规则：\n'
            '- Lact 与 Urea 不可报告，只能 reference\n'
            '- 若某目标的可报告性分级为 reject，不得 report\n'
            '- 你可以调用工具查询区间、分级、阈值、期望损失、退化状态\n'
            '- 理由码只能取：' + ', '.join(CONCEPT_CODES) + '\n'
            '最终必须输出 JSON：{"actions":{目标:动作},'
            '"reasons":{目标:[理由码]}}'
        )
        user_msg = ('目标列表：' + ', '.join(ctx['targets']) +
                    '\n请先查询必要信息，再给出决策。')

        messages = [{'role': 'system', 'content': sys_msg},
                    {'role': 'user', 'content': user_msg}]
        n_calls = 0

        for _round in range(self.MAX_ROUNDS):
            payload = {
                'model': self.model,
                'messages': messages,
                'tools': schema,
                'temperature': 0.0,
                'max_tokens': 700,
            }
            r = requests.post(f'{self.base_url}/chat/completions',
                              headers={'Authorization': f'Bearer {key}',
                                       'Content-Type': 'application/json'},
                              json=payload, timeout=self.timeout)
            r.raise_for_status()
            msg = r.json()['choices'][0]['message']
            calls = msg.get('tool_calls') or []

            if calls:
                messages.append({'role': 'assistant',
                                 'content': msg.get('content') or '',
                                 'tool_calls': calls})
                for c in calls:
                    fn = c['function']['name']
                    try:
                        args = json.loads(c['function'].get('arguments') or '{}')
                    except Exception:
                        args = {}
                    try:
                        out = tools.call(fn, args)
                    except Exception as e:
                        out = {'error': f'{type(e).__name__}: {e}'}
                    n_calls += 1
                    messages.append({'role': 'tool', 'tool_call_id': c['id'],
                                     'content': json.dumps(out,
                                                           ensure_ascii=False)})
                continue

            # 无 tool call ⇒ 期望最终 JSON
            content = msg.get('content') or ''
            obj = _parse_json_lenient(content)
            if obj is None:
                raise ValueError(f'LLM 未返回可解析 JSON：{content[:200]}')
            acts = obj.get('actions') or {}
            reasons = obj.get('reasons') or {}
            info['n_tool_calls'] = n_calls
            assert n_calls == len([m for m in messages
                                   if m.get('role') == 'tool']), \
                'n_tool_calls 与实际 tool 消息数不一致'
            return Proposal(actions={k: str(v) for k, v in acts.items()},
                            reasons={k: list(v) if isinstance(v, list) else [str(v)]
                                     for k, v in reasons.items()},
                            source='llm', n_tool_calls=n_calls, raw=content)

        raise RuntimeError(f'超过 {self.MAX_ROUNDS} 轮仍未给出决策')


def _parse_json_lenient(text):
    """从容地从中抽取 JSON 对象"""
    if not text:
        return None
    try:
        return json.loads(text)
    except Exception:
        pass
    m = re.search(r'\{.*\}', text, re.S)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            return None
    return None


# ---------------------------------------------------------------- 自测

def _mk_ctx():
    targets = ['Fat', 'Prot', 'Lact', 'TS', 'Urea']
    return {
        'targets': targets,
        'lo': np.array([3.95, 3.40, 4.80, 12.20, 20.0]),
        'hi': np.array([4.35, 3.60, 5.00, 12.60, 30.0]),
        'sigma': np.array([0.10, 0.05, 0.05, 0.10, 2.5]),
        'q': np.array([0.30, 0.12, 0.20, 0.40, 2.0]),
        'grades': {
            'Fat': {'code': 'report', 'reportable': True, 'rpd': 5.25, 'r2': 0.9638},
            'Prot': {'code': 'report', 'reportable': True, 'rpd': 6.13, 'r2': 0.9734},
            'Lact': {'code': 'reject', 'reportable': False, 'rpd': 1.59, 'r2': 0.6044},
            'TS': {'code': 'report', 'reportable': True, 'rpd': 5.81, 'r2': 0.9704},
            'Urea': {'code': 'reject', 'reportable': False, 'rpd': 1.07, 'r2': 0.1268},
        },
        'grade_thr': {'Fat': (4.0, 4.5, 5.0), 'Prot': (3.4, 3.7, 4.0),
                      'TS': (13.5, 14.5, 15.5)},
        'compliance': {'Fat': 3.1, 'Prot': 2.9, 'TS': None},
        'regret': {
            'Fat': {'report': 1.0, 'retest': 2.0, 'reference': 5.0},
            'Prot': {'report': 0.8, 'retest': 1.5, 'reference': 4.0},
            'TS': {'report': 1.2, 'retest': 2.2, 'reference': 6.0},
        },
        'degraded': False,
        'missing': [],
    }


def _self_test() -> int:
    ok, fail = 0, []
    print('=' * 74)
    print('a5_guarded.py 自测（受约束的 Agent）')
    print('=' * 74)

    ctx = _mk_ctx()
    tools = Tools(ctx)

    # ① 工具返回正确
    iv = tools.get_interval('Fat')
    if abs(iv['width'] - 0.40) < 1e-9 and iv['target'] == 'Fat':
        ok += 1
    else:
        fail.append(f'get_interval 异常 {iv}')

    # ② 未声明参数被过滤
    out = tools.call('get_interval', {'target': 'Fat', 'Y': [1, 2, 3]})
    if out.get('target') == 'Fat' and 'Y' not in out:
        ok += 1
    else:
        fail.append('工具未过滤未声明参数')

    # ③ 未知工具 / 未知目标报错
    try:
        tools.call('drop_table', {})
        fail.append('未知工具未报错')
    except KeyError:
        ok += 1
    try:
        tools.call('get_interval', {'target': 'NotATarget'})
        fail.append('未知目标未报错')
    except Exception:
        ok += 1

    # ④ 合法提议 ⇒ 通过
    good = Proposal(actions={'Fat': 'report', 'Prot': 'report', 'Lact': 'reference',
                             'TS': 'report', 'Urea': 'reference'},
                    reasons={'Fat': ['NONE'], 'Prot': ['THRESHOLD_CROSS'],
                             'Lact': ['LOW_RPD'], 'TS': ['NONE'],
                             'Urea': ['LOW_RPD']}, source='llm')
    g = Guard().check(good, ctx)
    print(f'  合法提议 → ok={g.ok}')
    if g.ok:
        ok += 1
    else:
        fail.append(f'合法提议被拒：{g.violations}')

    # ④b 引用**区间宽度**（工具暴露的派生量）必须通过 G5
    w = float(ctx['hi'][0] - ctx['lo'][0])
    prop_w = Proposal(actions=good.actions, reasons=good.reasons,
                      raw=f'Fat 区间宽度 {w:.4f}，可报告')
    gw = Guard().check(prop_w, ctx)
    print(f'  引用区间宽度 {w:.4f} → ok={gw.ok}'
          f'{"" if gw.ok else "  " + str(gw.violations)}')
    if gw.ok:
        ok += 1
    else:
        fail.append(f'引用工具暴露的 width 被判不可溯源：{gw.violations}')

    # ⑤ 阴性对照：逐条规则各造一个违规
    bad_cases = {
        'G1_valid_action': Proposal(
            actions={**good.actions, 'Fat': 'ship_it'}, reasons=good.reasons),
        'G1b_cover_targets': Proposal(
            actions={'Fat': 'report'}, reasons=good.reasons),
        'G2_no_report_nonreportable': Proposal(
            actions={**good.actions, 'Lact': 'report'}, reasons=good.reasons),
        'G4_valid_reason': Proposal(
            actions=good.actions,
            reasons={**good.reasons, 'Fat': ['MAGIC_REASON']}),
        'G5_numbers_traceable': Proposal(
            actions=good.actions, reasons=good.reasons,
            raw='结论：Fat 的误差约 0.1234567，很可靠'),
        'G6_not_worst_action': Proposal(
            actions={**good.actions, 'Fat': 'report'},
            reasons=good.reasons,
            raw=None),
    }
    for rule, prop in bad_cases.items():
        if rule == 'G6_not_worst_action':
            c2 = _mk_ctx()
            c2['regret']['Fat'] = {'report': 9.0, 'retest': 1.0, 'reference': 2.0}
            gg = Guard().check(prop, c2)
        else:
            gg = Guard().check(prop, ctx)
        hit = (not gg.ok) and (gg.checked.get(rule) is False)
        print(f'  阴性 {rule:28} → 检出={not gg.ok} '
              f'（该规则失败={gg.checked.get(rule)}）')
        if hit:
            ok += 1
        else:
            fail.append(f'阴性对照未命中：{rule}')

    # ⑥ G3：整族缺失时不得 report
    c3 = _mk_ctx()
    c3['degraded'] = True
    c3['missing'] = ['b7', 'b8', 'b9']
    g3 = Guard().check(good, c3)
    print(f'  退化（整族缺失）+ report → ok={g3.ok}')
    if not g3.ok and g3.checked.get('G3_no_report_degraded') is False:
        ok += 1
    else:
        fail.append('G3 未拦下整族缺失下的 report')

    # ⑦ 无密钥 ⇒ 纯规则回退
    had = os.environ.pop('LLM_API_KEY', None)
    try:
        ag = GuardedAgent()
        prop, gr, info = ag.decide(ctx)
        print(f'  无密钥 → source={prop.source}  fallback={info["fallback"]}')
        if prop.source == 'rule' and info['fallback']:
            ok += 1
        else:
            fail.append('无密钥时未回退到规则')
    finally:
        if had is not None:
            os.environ['LLM_API_KEY'] = had

    # ⑦b G7 预算门控：逐样本合法但整批超预算 ⇒ 必须裁剪
    ctxs = [_mk_ctx() for _ in range(10)]

    class _FakeLLM(GuardedAgent):
        def decide(self, ctx):
            p = Proposal(actions={t: ('retest' if t in ('Fat', 'Prot', 'TS')
                                     else 'reference') for t in ctx['targets']},
                         reasons={t: ['HIGH_SIGMA'] for t in ctx['targets']},
                         source='llm', n_tool_calls=5)
            return p, Guard().check(p, ctx), {'llm_used': True, 'guard_ok': True}

    fp, fd = _FakeLLM().decide_batch(ctxs, budget_frac=0.20)
    n_ret = sum(1 for p in fp if any(a == 'retest' for a in p.actions.values()))
    n_rule = sum(1 for p in fp if p.source == 'rule')
    print(f'  G7 预算门控：10 样本全 retest ⇒ 最终 retest {n_ret}（预算 2），'
          f'回退规则 {n_rule}')
    if n_ret <= 2 and n_rule >= 8:
        ok += 1
    else:
        fail.append(f'G7 预算门控失效：retest={n_ret}, rule={n_rule}')

    # ⑧ 规则提议本身通过门控
    ag = GuardedAgent()
    rp = ag._rule_proposal(ctx)
    rg = Guard().check(rp, ctx)
    print(f'  规则提议 → ok={rg.ok}  actions={rp.actions}')
    if rg.ok:
        ok += 1
    else:
        fail.append(f'规则提议被自己的门控拒绝：{rg.violations}')

    print(f'  通过 {ok} 项')
    if fail:
        print(f'  ❌ 失败 {len(fail)} 项：')
        for f_ in fail:
            print('     -', f_)
        return 1
    print('  ✅ 全部通过：工具只读且过滤参数；7 条门控规则均可用阴性对照推翻；'
          '无密钥时纯规则回退')
    return 0


if __name__ == '__main__':
    sys.exit(_self_test())
