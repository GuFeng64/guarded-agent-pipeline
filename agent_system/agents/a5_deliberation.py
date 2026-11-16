# -*- coding: utf-8 -*-
"""agents/a5_deliberation.py — **真多 Agent 交互**：agent-as-tool + 审议环

## 与 a5_guarded.py 的区别
    a5_guarded：**单** Agent（LLM 提议 + 确定性门控）
                它的 5 个工具只查**数据**，不调用其他 Agent
    a5_deliberation（本模块）：**真正的多 Agent**
                ① **agent-as-tool**：决策 Agent 可调用**感知 Agent（A1）**
                   请它复查光谱（尖峰/饱和/NaN）
                ② **审议环（proposer–critic）**：
                   A5b 提议 → V 审计 Agent **质疑** → A5b **修订** → 再审
                   最多 N 轮；仍不通过则回退规则

## 为什么这是"真多 Agent"
    · **Agent 间消息往返**（提议/质疑/修订，多轮）
    · **能力边界不同**：A5b 能决策但看不到光谱；A1 能看光谱但不能决策
      ⇒ 必须**跨 Agent 请求**才能获得该信息
    · **非中心化**：审议结果由两 Agent 的交互决定，不是编排器写死

## 判据（自测与评估都要满足）
    ① A5b 在区间宽时**主动**请求 A1 复查（工具调用计数 > 0）
    ② V 的质疑能**触发** A5b 修订（修订后通过）
    ③ 审议 N 轮仍不通过 ⇒ 回退规则（不劣化）
    ④ 全程不接触真值（A1 只看光谱，V 只审报告结构）
"""
from __future__ import annotations

import json
import os
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

from a5_guarded import (ACTION_CODES, CONCEPT_CODES, NON_REPORTABLE,
                        Guard, GuardResult, GuardedAgent, Proposal,
                        _parse_json_lenient)

__all__ = ['Critique', 'CriticAgent', 'MultiAgent', 'SpectrumTools']


# ---------------------------------------------------------------- 批判（V 的声音）

@dataclass
class Critique:
    ok: bool
    violations: List[str] = field(default_factory=list)
    hints: List[str] = field(default_factory=list)   # 给提议方的修订建议
    round: int = 0

    def __repr__(self):
        return f'Critique(ok={self.ok}, n_viol={len(self.violations)})'


class CriticAgent:
    """审计 Agent：对提议给出**结构化质疑**（而非仅仅拒绝）。

    它复用 `Guard` 的 7 条规则，但把结果转成**可执行的修订建议**。
    """

    #: 每条规则失败时给提议方的建议
    HINTS = {
        'G1_valid_action': '动作只能是 report / retest / reference',
        'G1b_cover_targets': '必须为每个目标给出一个动作',
        'G2_no_report_nonreportable': 'Lact 与 Urea 不可报告，只能 reference',
        'G3_no_report_degraded': '整族缺失去退化时不得 report',
        'G4_valid_reason': '理由码必须在允许集合内',
        'G5_numbers_traceable': '不要引用未通过工具查询到的数值',
        'G6_not_worst_action': '若 report 的期望损失最高，应改为 retest 或 reference',
    }

    def __init__(self, guard=None):
        self.guard = guard or Guard()

    def review(self, prop: Proposal, ctx: dict, rnd: int = 0) -> Critique:
        g = self.guard.check(prop, ctx)
        hints = []
        for rule, passed in g.checked.items():
            if not passed:
                hints.append(self.HINTS.get(rule, f'{rule} 未通过'))
        return Critique(ok=g.ok, violations=list(g.violations),
                        hints=hints, round=rnd)


# ---------------------------------------------------------------- A1 作为工具

class SpectrumTools:
    """把**感知 Agent（A1）**包装成决策 Agent 可调用的工具。

    ⚠️ 这是真"agent-as-tool"：决策 Agent 看不到光谱，必须**请求** A1。
    """

    def __init__(self, x_views: dict, try_repair=True):
        self.x_views = x_views
        self.try_repair = try_repair
        self._cache = None

    def request_spectral_recheck(self):
        """请 A1 复查该样本的光谱，返回结构化结论（不含真值）。"""
        if self._cache is not None:
            return self._cache
        from a1_acquisition import AcquisitionAgent
        A1 = AcquisitionAgent(try_repair=self.try_repair)
        r = A1.check_one(self.x_views)
        self._cache = {
            'agent': 'A1_acquisition',
            'verdict': r.verdict,
            'corrected': bool(r.corrected),
            'spike_index': float(r.spike_index),
            'reasons': list(r.reasons),
            'checks': dict(r.checks),
        }
        return self._cache

    def registry(self):
        return {'request_spectral_recheck': (self.request_spectral_recheck, {})}

    def call(self, name, args=None):
        reg = self.registry()
        if name not in reg:
            raise KeyError(f'未知工具 {name!r}；可用 {sorted(reg)}')
        return reg[name][0]()

    def schema(self):
        return [{'type': 'function', 'function': {
            'name': 'request_spectral_recheck',
            'description': '请感知 Agent（A1）复查该样本的光谱质量'
                           '（尖峰/饱和/NaN/缺视图）。返回 verdict 与原因。',
            'parameters': {'type': 'object', 'properties': {}, 'required': []},
        }}]


# ---------------------------------------------------------------- 多 Agent 主体

class MultiAgent:
    """审议式决策：A5b 提议 ⇄ V 质疑 ⇄ A5b 修订。"""

    MAX_ROUNDS = 3

    def __init__(self, model='deepseek-chat',
                 base_url='https://api.deepseek.com', timeout=60.0):
        self.base = GuardedAgent(model=model, base_url=base_url,
                                 timeout=timeout)
        self.critic = CriticAgent()

    @property
    def available(self):
        return self.base.available

    # ---- 主入口 ----
    def decide(self, ctx: dict, x_views: Optional[dict] = None,
               verbose=False) -> Tuple[Proposal, Critique, Dict]:
        """返回 (最终提议, 最终批判, 诊断)"""
        info = {'rounds': 0, 'a1_calls': 0, 'n_tool_calls': 0,
                'revisions': 0, 'fallback': False, 'llm_used': False,
                'transcript': []}
        rule = self.base._rule_proposal(ctx)
        info['rule_actions'] = dict(rule.actions)

        if not self.available:
            info['fallback'] = True
            info['reason'] = '无 LLM 密钥 ⇒ 纯规则'
            return rule, Critique(ok=True), info

        spec = SpectrumTools(x_views) if x_views else None
        prop = None
        crit = Critique(ok=True)

        for rnd in range(self.MAX_ROUNDS):
            info['rounds'] = rnd + 1
            try:
                if prop is None:
                    prop = self._propose(ctx, spec, info, critique=None)
                else:
                    prop = self._propose(ctx, spec, info, critique=crit)
                    info['revisions'] += 1
            except Exception as e:
                info['error'] = f'{type(e).__name__}: {e}'
                info['fallback'] = True
                return rule, crit, info

            info['llm_used'] = True
            crit = self.critic.review(prop, ctx, rnd)
            info['transcript'].append({
                'round': rnd + 1,
                'proposal': dict(prop.actions),
                'critique_ok': crit.ok,
                'violations': list(crit.violations),
            })
            if verbose:
                print(f'    [round {rnd + 1}] prop={prop.actions} '
                      f'⇒ critique ok={crit.ok} {crit.violations}')
            if crit.ok:
                info['accepted'] = True
                return prop, crit, info

        # N 轮仍不通过 ⇒ 回退
        info['fallback'] = True
        info['reason'] = f'审议 {self.MAX_ROUNDS} 轮仍未通过 ⇒ 回退规则'
        return rule, crit, info

    # ---- 一次提议（可带上一轮批判）----
    def _propose(self, ctx, spec, info, critique=None) -> Proposal:
        import requests
        key = os.environ['LLM_API_KEY']
        from a5_guarded import Tools
        tools = Tools(ctx)
        schemas = tools.schema()
        if spec is not None:
            schemas = schemas + spec.schema()

        sys_msg = (
            '你是乳品近红外检测的**决策 Agent**。为每个目标在 '
            'report / retest / reference 中选择一个动作。\n'
            '规则：\n'
            '- Lact 与 Urea 不可报告，只能 reference\n'
            '- 若某目标分级为 reject，不得 report\n'
            '- 若区间很宽，可调用 request_spectral_recheck 请**感知 Agent**'
            '复查光谱，再据此决定是否 retest\n'
            '- 可查询区间/分级/阈值/期望损失/退化状态\n'
            '- 理由码只能取：' + ', '.join(CONCEPT_CODES) + '\n'
            '- 若收到**审计 Agent 的质疑**，必须逐条修正后重新给出决策\n'
            '最终输出 JSON：{"actions":{目标:动作},"reasons":{目标:[理由码]}}'
        )
        user_msg = '目标列表：' + ', '.join(ctx['targets'])
        if critique is not None and not critique.ok:
            user_msg += ('\n\n【审计 Agent 的质疑】\n'
                         + '\n'.join(f'- {v}' for v in critique.violations)
                         + '\n请据此修订决策。')

        messages = [{'role': 'system', 'content': sys_msg},
                    {'role': 'user', 'content': user_msg}]
        n_calls = 0

        for _r in range(self.base.MAX_ROUNDS):
            payload = {'model': self.base.model, 'messages': messages,
                       'tools': schemas, 'temperature': 0.0,
                       'max_tokens': 800}
            resp = requests.post(f'{self.base.base_url}/chat/completions',
                                 headers={'Authorization': f'Bearer {key}',
                                          'Content-Type': 'application/json'},
                                 json=payload, timeout=self.base.timeout)
            resp.raise_for_status()
            msg = resp.json()['choices'][0]['message']
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
                    out = None
                    if spec is not None and fn == 'request_spectral_recheck':
                        out = spec.call(fn, args)
                        info['a1_calls'] += 1
                    else:
                        try:
                            out = tools.call(fn, args)
                        except Exception as e:
                            out = {'error': f'{type(e).__name__}: {e}'}
                    n_calls += 1
                    messages.append({'role': 'tool', 'tool_call_id': c['id'],
                                     'content': json.dumps(out,
                                                           ensure_ascii=False)})
                continue
            content = msg.get('content') or ''
            obj = _parse_json_lenient(content)
            if obj is None:
                raise ValueError(f'LLM 未返回可解析 JSON：{content[:200]}')
            info['n_tool_calls'] += n_calls
            return Proposal(actions={k: str(v) for k, v in
                                     (obj.get('actions') or {}).items()},
                            reasons={k: (list(v) if isinstance(v, list) else [str(v)])
                                     for k, v in (obj.get('reasons') or {}).items()},
                            source='llm', n_tool_calls=n_calls, raw=content)
        raise RuntimeError(f'超过 {self.base.MAX_ROUNDS} 轮仍未给出决策')


# ---------------------------------------------------------------- 自测

def _mk_ctx_and_views():
    import mems_data as M
    d = M.load()
    RV = ('Multi SE14', 'Multi_new SE20', 'Reflection SE20', 'Multi SE25')
    views = {v: d['X'][v][0].astype(np.float64) for v in RV}
    targets = ['Fat', 'Prot', 'Lact', 'TS', 'Urea']
    ctx = {
        'targets': targets,
        'lo': np.array([3.95, 3.40, 4.80, 12.20, 20.0]),
        'hi': np.array([4.35, 3.60, 5.00, 12.60, 30.0]),
        'sigma': np.array([0.10, 0.05, 0.05, 0.10, 2.5]),
        'q': np.array([0.31, 0.11, 0.16, 0.35, 2.0]),
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
    return ctx, views


def _self_test() -> int:
    ok, fail = 0, []
    print('=' * 76)
    print('a5_deliberation.py 自测（多 Agent：agent-as-tool + 审议环）')
    print('=' * 76)

    ctx, views = _mk_ctx_and_views()

    # ① A1 作为工具可被调用，返回结构化结论
    spec = SpectrumTools(views)
    r = spec.call('request_spectral_recheck', {})
    print(f"  A1 复查结果：verdict={r['verdict']} corrected={r['corrected']} "
          f"spike_index={r['spike_index']:.4f}")
    if r['agent'] == 'A1_acquisition' and r['verdict'] in ('ACCEPT', 'RESHOOT'):
        ok += 1
    else:
        fail.append(f'A1 工具返回异常：{r}')

    # ② A1 工具结果被缓存（不重复计算）
    r2 = spec.call('request_spectral_recheck', {})
    if r2 is r:
        ok += 1
    else:
        fail.append('A1 工具未缓存')

    # ③ 未知工具报错
    try:
        spec.call('drop_table', {})
        fail.append('未知工具未报错')
    except KeyError:
        ok += 1

    # ④ Critic 对合法提议不质疑
    critic = CriticAgent()
    good = Proposal(actions={'Fat': 'report', 'Prot': 'report',
                             'Lact': 'reference', 'TS': 'report',
                             'Urea': 'reference'},
                    reasons={'Fat': ['NONE'], 'Prot': ['NONE'],
                             'Lact': ['LOW_RPD'], 'TS': ['NONE'],
                             'Urea': ['LOW_RPD']})
    c0 = critic.review(good, ctx)
    print(f'  合法提议 → 质疑 ok={c0.ok}')
    if c0.ok:
        ok += 1
    else:
        fail.append(f'合法提议被质疑：{c0.violations}')

    # ⑤ Critic 对违规提议给出**可执行建议**
    bad = Proposal(actions={'Fat': 'report', 'Prot': 'report',
                            'Lact': 'report', 'TS': 'report',
                            'Urea': 'reference'},
                   reasons={'Fat': ['NONE'], 'Prot': ['NONE'],
                            'Lact': ['NONE'], 'TS': ['NONE'],
                            'Urea': ['LOW_RPD']})
    c1 = critic.review(bad, ctx)
    print(f'  Lact 被要求 report → 质疑 ok={c1.ok} 建议={c1.hints}')
    if (not c1.ok) and c1.hints and any('Lact' in h for h in c1.hints):
        ok += 1
    else:
        fail.append(f'质疑未给出可执行建议：{c1.hints}')

    # ⑥ 审议环：模拟"首轮违规、修订后通过"
    calls = {'n': 0}

    class _FakeMulti(MultiAgent):
        def _propose(self, ctx2, spec2, info, critique=None):
            calls['n'] += 1
            if critique is None:
                return Proposal(actions={'Fat': 'report', 'Prot': 'report',
                                         'Lact': 'report', 'TS': 'report',
                                         'Urea': 'reference'},
                                reasons={t: ['NONE'] for t in ctx2['targets']},
                                source='llm', n_tool_calls=3)
            # 收到质疑后修正
            return Proposal(actions=dict(good.actions),
                            reasons=dict(good.reasons),
                            source='llm', n_tool_calls=3)

    os.environ['LLM_API_KEY'] = 'fake-for-test'
    try:
        ma = _FakeMulti()
        prop, crit, info = ma.decide(ctx, views, verbose=False)
        print(f"  审议环：轮数={info['rounds']} 修订={info['revisions']} "
              f"接受={info.get('accepted')} 回退={info['fallback']}")
        if (info['rounds'] == 2 and info['revisions'] == 1
                and info.get('accepted') and prop.actions == good.actions):
            ok += 1
        else:
            fail.append(f'审议环未按预期收敛：{info}')
    finally:
        os.environ.pop('LLM_API_KEY', None)

    # ⑦ 审议 N 轮不通过 ⇒ 回退规则
    class _Stubborn(MultiAgent):
        def _propose(self, ctx2, spec2, info, critique=None):
            return Proposal(actions={'Fat': 'report', 'Prot': 'report',
                                     'Lact': 'report', 'TS': 'report',
                                     'Urea': 'reference'},
                            reasons={t: ['NONE'] for t in ctx2['targets']},
                            source='llm', n_tool_calls=2)

    os.environ['LLM_API_KEY'] = 'fake-for-test'
    try:
        ma2 = _Stubborn()
        prop2, crit2, info2 = ma2.decide(ctx, views)
        print(f"  顽固提议：轮数={info2['rounds']} 回退={info2['fallback']} "
              f"source={prop2.source}")
        if info2['fallback'] and prop2.source == 'rule':
            ok += 1
        else:
            fail.append(f'顽固提议未回退规则：{info2}')
    finally:
        os.environ.pop('LLM_API_KEY', None)

    # ⑧ 无密钥 ⇒ 纯规则
    ma3 = MultiAgent()
    prop3, crit3, info3 = ma3.decide(ctx, views)
    if prop3.source == 'rule' and info3['fallback']:
        ok += 1
    else:
        fail.append('无密钥时未回退')

    print(f'  通过 {ok} 项')
    if fail:
        print(f'  ❌ 失败 {len(fail)} 项：')
        for f_ in fail:
            print('     -', f_)
        return 1
    print('  ✅ 全部通过：A1 可作为工具被调用且缓存；质疑带可执行建议；'
          '审议环 2 轮收敛；顽固提议回退规则；无密钥纯规则')
    return 0


if __name__ == '__main__':
    sys.exit(_self_test())
