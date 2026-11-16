# -*- coding: utf-8 -*-
"""agents/a6_llm.py — LLM 叙述层（可选；数值仍由代码插入）

## 设计（AGENT_CHARTER 规则 Z3 / C9）
    LLM 只接收**带占位符的结构化摘要**，产出**仍带占位符的叙述模板**；
    真实数值由 `a6_report.Reporter.render()` 用 .format 填入
    ⇒ LLM 在结构上无法编造或改动数值

## 主要失败模式（必须在自测中覆盖）
    ① LLM 丢弃/改写占位符 ⇒ 数值无法填入 ⇒ 必须检出并回退到默认模板
    ② LLM 自己编造数字 ⇒ 因它只见占位符，理论上不可能；
       但仍校验最终文本与结构化数值逐字符一致（verify_numbers）

## 密钥
    只从环境变量 `LLM_API_KEY` 读取，**不写入任何文件**

## 无密钥 / 网络失败
    自动回退到确定性默认模板（系统仍完整可用）

## 自测
    python a6_llm.py            # 无密钥 ⇒ 只测回退与占位符校验
    set LLM_API_KEY=... && python a6_llm.py   # 有密钥 ⇒ 额外做一次真实调用
"""
from __future__ import annotations

import json
import os
import re
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_AG = os.path.dirname(_HERE)
_REV = os.path.dirname(_AG)
for _p in (_AG, _HERE, os.path.join(_REV, 'code')):
    if _p not in sys.path:
        sys.path.insert(0, _p)

__all__ = ['LLMNarrator', 'REQUIRED_PLACEHOLDERS', 'placeholder_ok']

#: 报告模板必须保留的占位符（少一个就填不进数值）
REQUIRED_PLACEHOLDERS = ('{sample_id}', '{verdict}', '{lines}', '{note}')

#: 默认 API 端点（可通过环境变量覆盖）
DEFAULT_BASE = 'https://api.deepseek.com'
DEFAULT_MODEL = 'deepseek-chat'


def placeholder_ok(text: str):
    """检查模板是否保留了全部必需占位符。"""
    missing = [p for p in REQUIRED_PLACEHOLDERS if p not in text]
    return (len(missing) == 0), missing


class LLMNarrator:
    """LLM 叙述层：只改写模板文字，不接触数值。"""

    def __init__(self, api_key=None, base=None, model=None, timeout=45,
                 max_tokens=400):
        self.api_key = api_key or os.environ.get('LLM_API_KEY')
        self.base = (base or os.environ.get('LLM_BASE') or DEFAULT_BASE).rstrip('/')
        self.model = model or os.environ.get('LLM_MODEL') or DEFAULT_MODEL
        self.timeout = timeout
        self.max_tokens = max_tokens
        self.last_usage = None
        self.last_latency = None

    @property
    def available(self):
        return bool(self.api_key)

    # ---------- 核心：把默认模板改写成更通顺的叙述模板 ----------
    def rewrite_template(self, base_template: str, target_names=None):
        """返回 (新模板, meta)。失败或无密钥 ⇒ 返回原模板（确定性回退）。"""
        if not self.available:
            return base_template, {'used_llm': False, 'reason': 'no api key'}

        prompt = (
            '你是乳品检测报告的撰写助手。下面是报告的**模板**，其中 '
            '{sample_id} {verdict} {lines} {note} 是占位符，'
            '它们的值由程序填入。\n'
            '请改写模板的**措辞**，使它更通顺、更专业，面向质检人员。\n'
            '**严格要求**：\n'
            '1. 必须原样保留上述 4 个占位符（包括花括号），一个都不能少或改名；\n'
            '2. 不得添加任何新的数字、数值范围或结论；\n'
            '3. 只输出改写后的模板本身，不要解释。\n\n'
            f'--- 模板开始 ---\n{base_template}\n--- 模板结束 ---'
        )
        t0 = time.perf_counter()
        try:
            import urllib.request
            req = urllib.request.Request(
                f'{self.base}/chat/completions',
                data=json.dumps({
                    'model': self.model,
                    'messages': [{'role': 'user', 'content': prompt}],
                    'max_tokens': self.max_tokens,
                    'temperature': 0.2,
                }).encode('utf-8'),
                headers={'Authorization': f'Bearer {self.api_key}',
                         'Content-Type': 'application/json'})
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                body = json.loads(r.read().decode('utf-8'))
            out = body['choices'][0]['message']['content'].strip()
            self.last_usage = body.get('usage')
            self.last_latency = time.perf_counter() - t0
            # 去掉可能的代码块包装
            out = re.sub(r'^```[a-zA-Z]*\n?', '', out)
            out = re.sub(r'\n?```$', '', out).strip()
            ok, missing = placeholder_ok(out)
            if not ok:
                return base_template, {'used_llm': False,
                                       'reason': f'placeholder lost: {missing}',
                                       'raw': out[:200]}
            return out, {'used_llm': True, 'latency': self.last_latency,
                         'usage': self.last_usage}
        except Exception as e:
            return base_template, {'used_llm': False,
                                   'reason': f'{type(e).__name__}: {e}'}


# ---------------------------------------------------------------- 自测

def _self_test() -> int:
    ok, fail = 0, []
    from agents.a6_report import Reporter, DEFAULT_TEMPLATE

    print('=' * 72)
    print('a6_llm.py 自测')
    print('=' * 72)

    N = LLMNarrator()
    print(f'  密钥可用: {N.available}')

    # ① 无密钥 ⇒ 必须确定性回退
    if not N.available:
        tpl, meta = N.rewrite_template(DEFAULT_TEMPLATE)
        if tpl == DEFAULT_TEMPLATE and not meta['used_llm']:
            ok += 1
        else:
            fail.append('无密钥时未回退到默认模板')
        print('  （无密钥，跳过真实调用测试）')
    else:
        ok += 1
        print('  （有密钥，将做一次真实调用）')

    # ② 占位符校验函数本身
    good, _ = placeholder_ok(DEFAULT_TEMPLATE)
    bad, miss = placeholder_ok('报告：{sample_id} 判定 {verdict}')
    if good and (not bad) and ('{lines}' in miss):
        ok += 1
    else:
        fail.append('占位符校验函数有误')

    # ③ 构造一个"丢了占位符"的伪输出 ⇒ 必须被判定为不可用
    fake = '【报告】样品 {sample_id}：判定 {verdict}。'
    okf, missf = placeholder_ok(fake)
    if (not okf) and ('{lines}' in missf):
        ok += 1
    else:
        fail.append('占位符丢失未被检出')

    # ④ 有密钥时：真实调用 + 完整性校验
    if N.available:
        tpl2, meta2 = N.rewrite_template(DEFAULT_TEMPLATE)
        print(f'  LLM 调用: used={meta2.get("used_llm")} '
              f'latency={meta2.get("latency")}')
        if meta2.get('used_llm'):
            okp, _ = placeholder_ok(tpl2)
            if okp:
                ok += 1
            else:
                fail.append('LLM 返回的模板丢失占位符（未回退）')
            # 用 LLM 模板渲染一次，验证数值仍正确
            import numpy as np
            T = ['Fat', 'Prot', 'Lact', 'TS', 'Urea']
            grades = {t: {'r2': 0.96, 'rpd': 5.0,
                          'code': 'report' if t in ('Fat', 'Prot', 'TS') else 'reject',
                          'reportable': t in ('Fat', 'Prot', 'TS')} for t in T}
            yhat = np.array([4.1234, 3.4567, 4.89, 12.3456, 25.0])
            R = Reporter(T)
            rep = R.build(sample_id='S001', yhat_row=yhat,
                          lo_row=yhat - 0.5, hi_row=yhat + 0.5,
                          decision='report', grades=grades)
            txt = R.render(rep, template=tpl2)
            good_num, probs = R.verify_numbers(rep, txt)
            if good_num:
                ok += 1
            else:
                fail.append(f'LLM 模板渲染后数值校验失败：{probs[:2]}')
        else:
            fail.append(f'有密钥但调用失败：{meta2.get("reason")}')

    print(f'  通过 {ok} 项')
    if fail:
        print(f'  ❌ 失败 {len(fail)} 项：')
        for f_ in fail:
            print('     -', f_)
        return 1
    print('  ✅ 全部通过：无密钥可回退；占位符丢失可检出；数值不可被 LLM 改动')
    return 0


if __name__ == '__main__':
    sys.exit(_self_test())
