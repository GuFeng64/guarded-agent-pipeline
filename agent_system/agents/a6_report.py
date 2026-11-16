# -*- coding: utf-8 -*-
"""agents/a6_report.py — 报告 Agent：结构化报告 + 模板渲染（LLM 可选且不得改数值）

## 核心设计（AGENT_CHARTER 规则 Z3）
    **数值链路中不得出现 LLM。**
    做法：LLM（若启用）只产出**带占位符的叙述模板**，
          数值由代码用 `str.format` 填入
    ⇒ LLM 在结构上**无法改动**任何数值

## 输入（上下文白名单：不含真值）
    yhat, lo, hi, decisions, grades, degraded, missing_branches, target_names

## 输出
    structured: dict   —— 全部数值（唯一权威）
    text:       str    —— 渲染后的报告

## 判据（自测）
    ① 关闭 LLM 时报告完整可读
    ② 报告中出现的每个数值都必须能在 structured 中找到（逐字符一致）
    ③ 模拟一个"试图篡改数值"的 LLM ⇒ 必须被检出（数值校验失败）

## 自测
    python a6_report.py
"""
from __future__ import annotations

import os
import re
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_AG = os.path.dirname(_HERE)
_REV = os.path.dirname(_AG)
for _p in (_AG, _HERE, os.path.join(_REV, 'code')):
    if _p not in sys.path:
        sys.path.insert(0, _p)

__all__ = ['Reporter', 'DEFAULT_TEMPLATE', 'extract_numbers']

#: 默认叙述模板（{key} 由此处的代码填入，LLM 只能改这段文字，不能改数值）
DEFAULT_TEMPLATE = (
    '【乳品 NIR 检测报告】样品 {sample_id}\n'
    '判定：{verdict}\n'
    '分项目结果：\n'
    '{lines}\n'
    '说明：{note}\n'
)

#: 单个目标的模板行
LINE_TEMPLATE = ('  · {target}：{value}（{interval}），R²={r2}，RPD={rpd}，'
                 '等级={grade}')


def extract_numbers(text: str):
    """抽取文本中的所有数值（含负数、小数）。"""
    return re.findall(r'-?\d+\.?\d*', text)


class Reporter:
    """报告 Agent 的实现。"""

    def __init__(self, targets, units=None):
        self.targets = list(targets)
        self.units = dict(units or {})

    # ---------- 结构化报告（唯一权威数值来源）----------
    def build(self, *, sample_id, yhat_row, lo_row, hi_row, decision,
              grades, degraded=False, missing=None, alpha=0.10):
        """构造结构化报告。yhat_row / lo_row / hi_row 为单样本一维数组。"""
        items = {}
        for t in self.targets:
            ti = self.targets.index(t)
            g = grades.get(t, {})
            items[t] = {
                'target': t,
                'value': float(yhat_row[ti]),
                'lo': float(lo_row[ti]),
                'hi': float(hi_row[ti]),
                'r2': float(g.get('r2', float('nan'))),
                'rpd': float(g.get('rpd', float('nan'))),
                'grade': str(g.get('code', 'unknown')),
                'reportable': bool(g.get('reportable', False)),
            }
        return {
            'sample_id': str(sample_id),
            'decision': str(decision),
            'interval_level': float(1 - alpha),
            'degraded': bool(degraded),
            'missing_branches': list(missing or []),
            'items': items,
        }

    # ---------- 渲染（模板；可选 LLM 只改文字）----------
    def render(self, report, template=None, narrative_fn=None):
        """渲染报告文本。narrative_fn 可替换叙述文字，但**必须**返回相同占位符。"""
        tpl = template or DEFAULT_TEMPLATE
        if narrative_fn is not None:
            tpl = narrative_fn(tpl)          # ⭐ LLM 只能改文字结构

        lines = []
        for t in self.targets:
            it = report['items'][t]
            if not it['reportable']:
                # 不可报告目标：只给等级，**不给数值**（防止误用）
                lines.append(f'  · {t}：不可用（等级={it["grade"]}），'
                             f'不提供数值，请转参考方法')
                continue
            lines.append(LINE_TEMPLATE.format(
                target=t,
                value=f'{it["value"]:.4f}',
                interval=f'{it["lo"]:.4f}~{it["hi"]:.4f}',
                r2=f'{it["r2"]:.4f}',
                rpd=f'{it["rpd"]:.2f}',
                grade=it['grade'],
            ))

        note = ('本报告含降级结果（缺失分支：'
                + (','.join(report['missing_branches']) or '无')
                + '）' if report['degraded'] else '分支完整')
        return tpl.format(sample_id=report['sample_id'],
                          verdict=report['decision'],
                          lines='\n'.join(lines),
                          note=note)

    # ---------- 数值溯源（白名单）----------
    def allowed_numbers(self, report):
        """构造报告中**允许出现**的数字白名单。

        ⚠️ 只允许**报告模板实际使用的格式**，否则舍入变体会让白名单过宽：
           例如 TS 值 12.3456 若允许 1 位小数，则 '12.3' 也能通过，
           编造"共 12.3 个批次"就逃过校验（已实测踩坑）。
        """
        allow = set()
        for t in self.targets:
            it = report['items'][t]
            if not it['reportable']:
                continue
            for key in ('value', 'lo', 'hi', 'r2'):
                v = it.get(key)
                if v is None or not np.isfinite(v):
                    continue
                allow.add(f'{float(v):.4f}')       # 模板用 4 位
            v = it.get('rpd')
            if v is not None and np.isfinite(v):
                allow.add(f'{float(v):.2f}')       # 模板用 2 位
        # 区间水平（如 90 / 95）—— 模板里是整数百分比
        lvl = float(report.get('interval_level', 0.9)) * 100
        allow.add(f'{lvl:.0f}')
        # 样本编号中的数字（原样）
        sid = str(report.get('sample_id', ''))
        for m in re.findall(r'\d+', sid):
            allow.add(m)
        return allow

    def verify_numbers_strict(self, report, text):
        """严格校验：文本中每个数字都必须能溯源。

        返回 (ok, problems)
        """
        allow = self.allowed_numbers(report)
        bad = []
        for tok in extract_numbers(text):
            if tok in ('', '-', '.'):
                continue
            # 规范化：去掉前导 + / 末尾 .
            t = tok.lstrip('+').rstrip('.')
            try:
                v = float(t)
            except ValueError:
                continue
            if t in allow:
                continue
            # 允许整数的 0/1/2/3 等小数字（"3 次""第 1 项"这类序数）
            if v in (0.0, 1.0, 2.0, 3.0, 4.0, 5.0) and t == str(int(v)) \
                    and '.' not in t:
                continue
            bad.append(tok)
        return (len(bad) == 0), bad

    # ---------- 数值一致性校验 ----------
    def verify_numbers(self, report, text, ndigits=4):
        """检查报告文本中的数值是否与 structured 完全一致。

        返回 (ok, problems)
        ⭐ 任何被改动/缺失的关键数值都会被检出。
        """
        problems = []
        for t in self.targets:
            it = report['items'][t]
            if not it['reportable']:
                continue
            need = [f'{it["value"]:.{ndigits}f}',
                    f'{it["lo"]:.{ndigits}f}',
                    f'{it["hi"]:.{ndigits}f}',
                    f'{it["r2"]:.{ndigits}f}',
                    f'{it["rpd"]:.2f}']
            for s in need:
                if s not in text:
                    problems.append(f'{t}: 数值 {s} 未出现在报告中（可能被改动）')
        return (len(problems) == 0), problems


# ---------------------------------------------------------------- 自测

def _self_test() -> int:
    ok, fail = 0, []
    print('=' * 72)
    print('a6_report.py 自测')
    print('=' * 72)

    T = ['Fat', 'Prot', 'Lact', 'TS', 'Urea']
    grades = {
        'Fat': {'r2': 0.9638, 'rpd': 5.25, 'code': 'report', 'reportable': True},
        'Prot': {'r2': 0.9734, 'rpd': 6.13, 'code': 'report', 'reportable': True},
        'Lact': {'r2': 0.6044, 'rpd': 1.59, 'code': 'reject', 'reportable': False},
        'TS': {'r2': 0.9704, 'rpd': 5.81, 'code': 'report', 'reportable': True},
        'Urea': {'r2': 0.1268, 'rpd': 1.07, 'code': 'reject', 'reportable': False},
    }
    yhat = np.array([4.1234, 3.4567, 4.8901, 12.3456, 25.0])
    lo = yhat - 0.5
    hi = yhat + 0.5
    R = Reporter(T)

    rep = R.build(sample_id='S001', yhat_row=yhat, lo_row=lo, hi_row=hi,
                  decision='report', grades=grades)

    # ① 无 LLM 渲染完整可读
    text = R.render(rep)
    if ('S001' in text) and ('Fat' in text) and ('TS' in text):
        ok += 1
    else:
        fail.append('无 LLM 渲染不完整')

    # ② 不可报告目标（Lact/Urea）不得出现数值
    no_val = ('Lact：不可用' in text) and ('Urea：不可用' in text)
    if no_val:
        ok += 1
    else:
        fail.append('不可报告目标仍在报告中给出数值')

    # ③ 数值一致性校验通过
    good, probs = R.verify_numbers(rep, text)
    if good:
        ok += 1
    else:
        fail.append(f'数值一致性校验失败：{probs[:3]}')

    # ④ 威胁模型：只有**对最终文本后处理**才能改数值
    #    ⭐ narrative_fn 只作用于外层模板，而 {value} 在 LINE_TEMPLATE 里
    #       ⇒ 模板层在结构上无法篡改数值；真正的风险是后处理
    text_bad = text.replace('4.1234', '9.9999')
    bad_ok, bad_probs = R.verify_numbers(rep, text_bad)
    if not bad_ok:
        ok += 1
    else:
        fail.append('后处理篡改数值未被检出')
    print('  ⭐ 威胁模型：LLM 只做模板改写 ⇒ 结构上无法改数值；'
          '只有对最终文本后处理才可能篡改 ⇒ 由 verify_numbers 兜底')

    # ④b 编造数值 ⇒ 严格校验必须检出
    fake = text.replace('说明：分支完整',
                        '说明：本批误差约 0.8765，建议复测 3 次')
    ok_s, bad_s = R.verify_numbers_strict(rep, fake)
    if not ok_s and any('0.8765' in b for b in bad_s):
        ok += 1
    else:
        fail.append(f'编造数值未被严格校验检出（bad={bad_s}）')

    # ④c 正常文本 ⇒ 严格校验应通过
    ok_s2, bad_s2 = R.verify_numbers_strict(rep, text)
    if ok_s2:
        ok += 1
    else:
        fail.append(f'正常文本被严格校验误判（bad={bad_s2}）')

    # ⑤ 正常的 LLM 改写文字（不改占位符）⇒ 仍通过
    def benign_llm(tpl):
        return tpl.replace('【乳品 NIR 检测报告】', '【质检报告（自动生成）】')

    text_ok2 = R.render(rep, narrative_fn=benign_llm)
    ok2, _ = R.verify_numbers(rep, text_ok2)
    if ok2:
        ok += 1
    else:
        fail.append('良性改写的 LLM 被误判为篡改')

    # ⑥ 降级标注
    rep2 = R.build(sample_id='S002', yhat_row=yhat, lo_row=lo, hi_row=hi,
                   decision='retest', grades=grades, degraded=True,
                   missing=['b8'])
    t2 = R.render(rep2)
    if ('降级' in t2) and ('b8' in t2):
        ok += 1
    else:
        fail.append('降级结果未被标注')

    print(f'  报告样例：\n{text}')
    print(f'  通过 {ok} 项')
    if fail:
        print(f'  ❌ 失败 {len(fail)} 项：')
        for f_ in fail:
            print('     -', f_)
        return 1
    print('  ✅ 全部通过：无 LLM 可用；不可报告目标无数值；'
          '篡改与**编造数值**均被检出；降级有标注')
    return 0


if __name__ == '__main__':
    sys.exit(_self_test())
