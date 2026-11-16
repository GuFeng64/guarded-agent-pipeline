# -*- coding: utf-8 -*-
"""agents/a7_feedback.py — 回流 Agent：标记高价值样本（**设计，未验证**）

## 职责
    把"被拒识（需复测）且已由参考法确认"的样本标记为高价值，
    建议纳入下一轮训练集（主动学习）。

## ⚠️ 诚实说明（AGENT_CHARTER 规则 Z2 / AGENT_SYSTEM 局限 ③）
    **本环节未验证**：没有"加入新样本后重训"的数据。
    ⇒ 论文中只能作为**设计**陈述，不得报告任何增益数字。

## 上下文隔离（规则 A1 / A4）
    只接收 validator 给出的 `confirmed_sample_ids`（已确认的样本 ID），
    **不接触任何真值本身**。

## 自测
    python a7_feedback.py
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from typing import List, Sequence

_HERE = os.path.dirname(os.path.abspath(__file__))
_AG = os.path.dirname(_HERE)
_REV = os.path.dirname(_AG)
for _p in (_AG, _HERE, os.path.join(_REV, 'code')):
    if _p not in sys.path:
        sys.path.insert(0, _p)

__all__ = ['FeedbackAgent', 'FeedbackResult']

#: ⚠️ 未验证声明（必须随任何输出一起呈现）
UNVALIDATED = ('回流环节**未经实验验证**（无新增样本重训数据）；'
               '本文仅作为系统设计陈述，不报告增益')


@dataclass
class FeedbackResult:
    high_value_ids: List[int] = field(default_factory=list)
    n_candidates: int = 0
    note: str = UNVALIDATED

    def __repr__(self):
        return (f'FeedbackResult(n_selected={len(self.high_value_ids)}, '
                f'candidates={self.n_candidates})')


class FeedbackAgent:
    """回流 Agent 的实现（规则型）。"""

    def __init__(self, min_score=None):
        self.min_score = min_score

    def select(self, decisions: Sequence[str],
               scores: Sequence[float],
               confirmed_ids: Sequence[int]) -> FeedbackResult:
        """挑选"需复测 + 已由参考法确认 + 得分高"的样本。

        decisions   : 每个样本的决策（'report' / 'retest'）
        scores      : 每个样本的规划得分（越高越不确定）
        confirmed_ids: validator 确认过的样本 ID（**只有 ID，无真值**）
        """
        conf = set(int(i) for i in confirmed_ids)
        picked = []
        for i, (d, s) in enumerate(zip(decisions, scores)):
            if d != 'retest':
                continue
            if i not in conf:
                continue
            if self.min_score is not None and s < self.min_score:
                continue
            picked.append(int(i))
        return FeedbackResult(high_value_ids=sorted(picked),
                              n_candidates=len(conf))


# ---------------------------------------------------------------- 自测

def _self_test() -> int:
    ok, fail = 0, []
    print('=' * 72)
    print('a7_feedback.py 自测')
    print('=' * 72)

    A = FeedbackAgent()
    decisions = ['report', 'retest', 'retest', 'report', 'retest']
    scores = [0.0, 3.0, 1.0, 0.0, 2.0]
    confirmed = [1, 4]          # 只有 1 和 4 被参考法确认

    r = A.select(decisions, scores, confirmed)
    # ① 只选"需复测且已确认"的样本
    if r.high_value_ids == [1, 4]:
        ok += 1
    else:
        fail.append(f'选择错误：{r.high_value_ids}，期望 [1, 4]')

    # ② 未确认的 retest 样本不得入选
    if 2 not in r.high_value_ids:
        ok += 1
    else:
        fail.append('未确认的样本被选入')

    # ③ report 样本不得入选
    if 0 not in r.high_value_ids and 3 not in r.high_value_ids:
        ok += 1
    else:
        fail.append('report 样本被选入')

    # ④ 必须带"未验证"声明
    if '未经实验验证' in r.note:
        ok += 1
    else:
        fail.append('缺少未验证声明')

    # ⑤ min_score 过滤
    A2 = FeedbackAgent(min_score=1.5)
    r2 = A2.select(decisions, scores, confirmed)
    if r2.high_value_ids == [1, 4]:
        ok += 1
    else:
        fail.append(f'min_score 过滤错误：{r2.high_value_ids}')

    # ⑥ 空确认集
    r3 = A.select(decisions, scores, [])
    if r3.high_value_ids == []:
        ok += 1
    else:
        fail.append('空确认集却选出样本')

    print(f'  ⚠️ {UNVALIDATED}')
    print(f'  通过 {ok} 项')
    if fail:
        print(f'  ❌ 失败 {len(fail)} 项：')
        for f_ in fail:
            print('     -', f_)
        return 1
    print('  ✅ 全部通过：只选"需复测且已确认"的样本；未验证声明随输出')
    return 0


if __name__ == '__main__':
    sys.exit(_self_test())
