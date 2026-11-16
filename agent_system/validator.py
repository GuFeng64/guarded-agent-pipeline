# -*- coding: utf-8 -*-
"""validator.py — 报告校验器：**唯一允许访问真值的环节**

## 职责（AGENT_CHARTER 规则 A1）
    只有它能看 Y / cow；其余 Agent 一律被 context.py 拦截。

## 校验项
    ① **区间覆盖**：真值是否落在报告的区间内（统计覆盖率是否达标）
    ② **等级合规**：报告中的 RPD 是否与该目标实际 RPD 相符
       ⭐ 能检出"等级被抬高"（例如把 Lact 报成 report）
    ③ **不可报告目标**：Lact/Urea 是否被报出了数值（必须没有）
    ④ **降级标注**：degraded=True 时报告是否写明缺失分支
    ⑤ **数值真伪**：报告中的点预测是否与该样本的实际预测一致（防篡改）

## 判据（自测）
    对人为注入的 5 类错误报告，必须全部检出。

## 自测
    python validator.py
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REV = os.path.dirname(_HERE)
for _p in (_HERE, os.path.join(_REV, 'code')):
    if _p not in sys.path:
        sys.path.insert(0, _p)

__all__ = ['Validator', 'ValidationReport']


@dataclass
class ValidationReport:
    ok: bool
    checks: Dict[str, bool] = field(default_factory=dict)
    problems: List[str] = field(default_factory=list)

    def add(self, name, passed, msg=None):
        self.checks[name] = bool(passed)
        if not passed:
            self.ok = False
            if msg:
                self.problems.append(msg)

    def __repr__(self):
        bad = [k for k, v in self.checks.items() if not v]
        return (f'ValidationReport(ok={self.ok}, checks={len(self.checks)}, '
                f'failed={bad})')


class Validator:
    """校验器（唯一可见真值）。"""

    def __init__(self, Y, cow, targets, rpd_of=None, reportable=None,
                 cov_target=0.90, cov_tol=0.05):
        self.Y = Y
        self.cow = cow
        self.targets = list(targets)
        self.ix = {t: i for i, t in enumerate(self.targets)}
        self.rpd_of = rpd_of or {}
        self.reportable = set(reportable or ['Fat', 'Prot', 'TS'])
        self.cov_target = cov_target
        self.cov_tol = cov_tol

    # ---------- ① 区间覆盖 ----------
    def check_coverage(self, rep, Y_row, row_idx):
        vr = ValidationReport(ok=True)
        for t in self.targets:
            it = rep['items'][t]
            if not it['reportable']:
                continue
            y = float(Y_row[self.ix[t]])
            inside = (it['lo'] - 1e-9) <= y <= (it['hi'] + 1e-9)
            vr.add(f'coverage_{t}', True)      # 单样本只记录，不判定
            if not inside:
                vr.problems.append(f'{t}: 真值 {y:.4f} 不在区间 '
                                   f'[{it["lo"]:.4f},{it["hi"]:.4f}] 内')
        return vr

    # ---------- ② 等级合规 ----------
    def check_grades(self, rep):
        vr = ValidationReport(ok=True)
        for t in self.targets:
            it = rep['items'][t]
            got = str(it['grade'])
            if t in self.reportable:
                vr.add(f'grade_{t}', got in ('report', 'good'),
                       f'{t}: 可报告目标被标为 {got!r}')
            else:
                vr.add(f'grade_{t}', got == 'reject',
                       f'{t}: 不可报告目标被标为 {got!r}（等级被抬高）')
        return vr

    # ---------- ③ 不可报告目标不得给数值 ----------
    def check_no_value_for_nonreportable(self, text):
        vr = ValidationReport(ok=True)
        for t in self.targets:
            if t in self.reportable:
                continue
            # 期望形态："{t}：不可用（等级=..."
            marker = f'{t}：不可用'
            vr.add(f'novalue_{t}', marker in text,
                   f'{t}: 报告中未声明"不可用"（可能给出了数值）')
        return vr

    # ---------- ④ 降级标注 ----------
    def check_degraded(self, rep, text):
        vr = ValidationReport(ok=True)
        if rep.get('degraded'):
            miss = rep.get('missing_branches') or []
            ok = ('降级' in text) and all(m in text for m in miss)
            vr.add('degraded_annotated', ok,
                   '降级结果未在报告中标注缺失分支')
        else:
            vr.add('degraded_annotated', True)
        return vr

    # ---------- ⑤ 数值真伪（防篡改）----------
    def check_values(self, rep, yhat_row, ndigits=4):
        vr = ValidationReport(ok=True)
        for t in self.targets:
            it = rep['items'][t]
            if not it['reportable']:
                continue
            exp = float(yhat_row[self.ix[t]])
            ok = abs(it['value'] - exp) < 1e-6
            vr.add(f'value_{t}', ok,
                   f'{t}: 报告值 {it["value"]:.6f} ≠ 实际预测 {exp:.6f}')
        return vr

    # ---------- 汇总 ----------
    # ---------- ⑥ 文本 vs 结构化报告的数值一致性 ----------
    def check_text_numbers(self, rep, text):
        """**文本**中的数值是否忠实于结构化报告。

        ⚠️ 与 `check_values` 互补：
           `check_values` 检查"结构化报告 vs 模型预测"；
           本方法检查"文本 vs 结构化报告"。
           用户拿到的是文本 ⇒ 后者同样是防篡改的必要环节。
        """
        vr = ValidationReport(ok=True)
        try:
            from agents.a6_report import Reporter
            R = Reporter(self.targets)
            ok1, probs1 = R.verify_numbers(rep, text)
            vr.add('text_numbers_present', ok1,
                   f'文本缺少报告中的数值：{probs1[:5]}')
            ok2, probs2 = R.verify_numbers_strict(rep, text)
            vr.add('text_numbers_no_fabrication', ok2,
                   f'文本出现无法溯源的数值：{probs2[:5]}')
        except Exception as e:
            vr.add('text_numbers_error', False, f'文本数值校验异常：{e}')
        return vr

    def validate(self, rep, text, row_idx, yhat_row, Y_row=None):
        Y_row = self.Y[row_idx] if Y_row is None else Y_row
        out = ValidationReport(ok=True)
        for sub in (self.check_coverage(rep, Y_row, row_idx),
                    self.check_grades(rep),
                    self.check_no_value_for_nonreportable(text),
                    self.check_degraded(rep, text),
                    self.check_values(rep, yhat_row),
            self.check_text_numbers(rep, text)):
            out.checks.update(sub.checks)
            out.problems.extend(sub.problems)
            out.ok = out.ok and sub.ok
        return out


# ---------------------------------------------------------------- 自测

def _self_test() -> int:
    ok, fail = 0, []
    from agents.a6_report import Reporter

    T = ['Fat', 'Prot', 'Lact', 'TS', 'Urea']
    grades = {
        'Fat': {'r2': 0.9638, 'rpd': 5.25, 'code': 'report', 'reportable': True},
        'Prot': {'r2': 0.9734, 'rpd': 6.13, 'code': 'report', 'reportable': True},
        'Lact': {'r2': 0.6044, 'rpd': 1.59, 'code': 'reject', 'reportable': False},
        'TS': {'r2': 0.9704, 'rpd': 5.81, 'code': 'report', 'reportable': True},
        'Urea': {'r2': 0.1268, 'rpd': 1.07, 'code': 'reject', 'reportable': False},
    }
    yhat = np.array([4.1234, 3.4567, 4.8901, 12.3456, 25.0])
    Y_row = np.array([4.05, 3.50, 4.80, 12.30, 26.0])     # 真值，在区间内
    rep_ok = {'sample_id': 'S1', 'decision': 'report', 'interval_level': 0.9,
              'degraded': False, 'missing_branches': [], 'items': {}}
    for t in T:
        ti = T.index(t)
        g = grades[t]
        rep_ok['items'][t] = {'target': t, 'value': float(yhat[ti]),
                              'lo': float(yhat[ti] - 0.5),
                              'hi': float(yhat[ti] + 0.5),
                              'r2': g['r2'], 'rpd': g['rpd'],
                              'grade': g['code'], 'reportable': g['reportable']}

    R = Reporter(T)
    text_ok = R.render(rep_ok)
    V = Validator(np.tile(Y_row, (3, 1)), np.array([1, 1, 2]), T,
                  reportable=['Fat', 'Prot', 'TS'])
    print('=' * 72)
    print('validator.py 自测（注入 5 类错误报告，必须全部检出）')
    print('=' * 72)

    # ① 正常报告应通过
    v0 = V.validate(rep_ok, text_ok, 0, yhat)
    if v0.ok:
        ok += 1
    else:
        fail.append(f'正常报告被误判：{v0.problems}')

    # ② 篡改数值 ⇒ 检出
    import copy
    r1 = copy.deepcopy(rep_ok)
    r1['items']['Fat']['value'] = 9.9999
    v1 = V.validate(r1, R.render(r1), 0, yhat)
    if not v1.ok:
        ok += 1
    else:
        fail.append('篡改数值未被检出')

    # ③ 等级被抬高（Lact → report）⇒ 检出
    r2 = copy.deepcopy(rep_ok)
    r2['items']['Lact']['grade'] = 'report'
    r2['items']['Lact']['reportable'] = True
    v2 = V.validate(r2, R.render(r2), 0, yhat)
    if not v2.ok:
        ok += 1
    else:
        fail.append('等级被抬高未被检出')

    # ④ 降级未标注 ⇒ 检出
    r3 = copy.deepcopy(rep_ok)
    r3['degraded'] = True
    r3['missing_branches'] = ['b8']
    txt3 = R.render(r3).replace('降级', '正常').replace('b8', '')
    v3 = V.validate(r3, txt3, 0, yhat)
    if not v3.ok:
        ok += 1
    else:
        fail.append('降级未标注未被检出')

    # ⑤ 不可报告目标被给了数值 ⇒ 检出
    txt4 = text_ok.replace('Lact：不可用（等级=reject），不提供数值，请转参考方法',
                           'Lact：4.8901（等级=reject）')
    v4 = V.validate(rep_ok, txt4, 0, yhat)
    if not v4.ok:
        ok += 1
    else:
        fail.append('不可报告目标给数值未被检出')

    # ⑥ 真值落在区间外 ⇒ 记录问题
    Y_out = Y_row.copy()
    Y_out[0] = yhat[0] + 5.0
    v5 = V.validate(rep_ok, text_ok, 0, yhat, Y_row=Y_out)
    if any('不在区间' in p for p in v5.problems):
        ok += 1
    else:
        fail.append('区间覆盖问题未被记录')

    print(f'  通过 {ok} 项')
    if fail:
        print(f'  ❌ 失败 {len(fail)} 项：')
        for f_ in fail:
            print('     -', f_)
        return 1
    print('  ✅ 全部通过：正常报告通过；5 类错误报告全部检出')
    return 0


if __name__ == '__main__':
    sys.exit(_self_test())
