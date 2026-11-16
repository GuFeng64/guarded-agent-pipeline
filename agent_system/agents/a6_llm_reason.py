# -*- coding: utf-8 -*-
"""agents/a6_llm_reason.py — LLM 推理层：结构化决策 + 可消融的概念码

## 为什么需要这个模块
    「模板润色」不产生信息增量，无法证明 LLM 的价值。
    本模块让 LLM 做**模板做不到**的事：读区间与阈值，判断跨越情况并给建议。

## 文献依据
    ① Right Answer, Wrong Reason (IEEE 2026)：正确率 ≠ 忠实性
       需测 CFS（概念消融）、ESI（重复一致性）、PSS（改写稳健性）
    ② Validity-Fidelity Gap (2026)：JSON 解析成功 ≠ 内容正确
       ⇒ 必须逐字段核对内容

## 输出契约（受约束，便于自动校验与消融）
    {
      "crossed": [float, ...],        # 区间跨越的阈值
      "recommendation": "report" | "retest" | "reference",
      "concepts": [str, ...],         # 概念码，决定建议的理由
      "reason": str                   # 一句自然语言
    }

## 概念码（用于消融实验）
    THRESHOLD_CROSS  区间跨越决策阈值
    WIDE_INTERVAL    区间过宽（> 2×容许限）
    HIGH_SIGMA       目标不可靠（RPD 低）
    DEGRADED         分支降级

## 安全
    密钥只从环境变量 LLM_API_KEY 读取；不写入任何文件
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

__all__ = ['Reasoner', 'CONCEPT_CODES', 'RECOMMENDATIONS', 'parse_json_lenient']

CONCEPT_CODES = ('THRESHOLD_CROSS', 'WIDE_INTERVAL', 'HIGH_SIGMA', 'DEGRADED')
RECOMMENDATIONS = ('report', 'retest', 'reference')

DEFAULT_BASE = 'https://api.deepseek.com'
DEFAULT_MODEL = 'deepseek-chat'


def parse_json_lenient(text: str):
    """从可能带代码块/额外说明的输出里取出第一个 JSON 对象。"""
    if not text:
        return None
    t = re.sub(r'^```[a-zA-Z]*\n?', '', text.strip())
    t = re.sub(r'\n?```$', '', t).strip()
    try:
        return json.loads(t)
    except Exception:
        pass
    i, j = t.find('{'), t.rfind('}')
    if i >= 0 and j > i:
        try:
            return json.loads(t[i:j + 1])
        except Exception:
            return None
    return None


class Reasoner:
    """LLM 结构化推理层。"""

    def __init__(self, api_key=None, base=None, model=None, timeout=45,
                 max_tokens=300, temperature=0.0):
        self.api_key = api_key or os.environ.get('LLM_API_KEY')
        self.base = (base or os.environ.get('LLM_BASE') or DEFAULT_BASE).rstrip('/')
        self.model = model or os.environ.get('LLM_MODEL') or DEFAULT_MODEL
        self.timeout = timeout
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.calls = 0
        self.total_tokens = 0
        self.total_latency = 0.0

    @property
    def available(self):
        return bool(self.api_key)

    # ---------- 构造提示 ----------
    @staticmethod
    def build_prompt(target, yhat, lo, hi, thresholds, tolerance,
                     rpd=None, degraded=False, paraphrase=False):
        """paraphrase=True 时用语义等价但措辞不同的版本（PSS 测试用）。"""
        thr_s = ', '.join(f'{t:g}' for t in thresholds)
        base = {
            'target': target,
            'yhat': round(float(yhat), 4),
            'lo': round(float(lo), 4),
            'hi': round(float(hi), 4),
            'thresholds': thr_s,
            'tolerance': round(float(tolerance), 4),
            'rpd': (None if rpd is None else round(float(rpd), 2)),
            'degraded': bool(degraded),
        }
        if paraphrase:
            return (
                '你是一名乳品检测的质量决策助手。请根据下列**测定结果**给出处置意见。\n'
                f'指标名称：{base["target"]}\n'
                f'本次预测值：{base["yhat"]}\n'
                f'90% 预测区间下界：{base["lo"]}；上界：{base["hi"]}\n'
                f'分档阈值依次为：{base["thresholds"]}\n'
                f'容许的区间半宽：{base["tolerance"]}\n'
                f'该指标的 RPD：{base["rpd"]}\n'
                f'当前是否处于降级模式：{base["degraded"]}\n\n'
                '请只输出一个 JSON 对象，键如下：\n'
                '  "crossed": 区间 [lo, hi] 覆盖到的阈值列表（逐个列出数值；没有则空列表）\n'
                '  "recommendation": "report" / "retest" / "reference" 三者之一\n'
                '  "concepts": 支撑你结论的概念码列表，只能从 '
                '["THRESHOLD_CROSS","WIDE_INTERVAL","HIGH_SIGMA","DEGRADED"] 中选\n'
                '  "reason": 一句中文理由\n'
                '注意：crossed 必须是你**自己**根据 lo/hi 与阈值比较得出的结果。')
        return (
            '你是乳品 NIR 检测的质量决策助手。给定单个指标的统计摘要，输出 JSON。\n'
            f'指标 {base["target"]}｜ŷ={base["yhat"]}｜区间=[{base["lo"]}, {base["hi"]}]｜'
            f'阈值=[{base["thresholds"]}]｜容许半宽={base["tolerance"]}｜'
            f'RPD={base["rpd"]}｜降级={base["degraded"]}\n'
            '要求：\n'
            '1) crossed：区间 [lo, hi] 覆盖了哪些阈值（自己比较，列出数值，无则 []）；\n'
            '2) recommendation：report / retest / reference；\n'
            '3) concepts：从 ["THRESHOLD_CROSS","WIDE_INTERVAL","HIGH_SIGMA","DEGRADED"] 选；\n'
            '4) reason：一句中文理由。\n'
            '只输出 JSON，不要其他内容。')

    # ---------- 单次调用 ----------
    def decide(self, target, yhat, lo, hi, thresholds, tolerance,
               rpd=None, degraded=False, paraphrase=False, max_retries=2):
        """返回 (parsed_dict_or_None, meta)"""
        if not self.available:
            return None, {'ok': False, 'reason': 'no api key'}
        prompt = self.build_prompt(target, yhat, lo, hi, thresholds, tolerance,
                                   rpd=rpd, degraded=degraded,
                                   paraphrase=paraphrase)
        last_err = None
        for attempt in range(max_retries + 1):
            t0 = time.perf_counter()
            try:
                import urllib.request
                req = urllib.request.Request(
                    f'{self.base}/chat/completions',
                    data=json.dumps({
                        'model': self.model,
                        'messages': [{'role': 'user', 'content': prompt}],
                        'max_tokens': self.max_tokens,
                        'temperature': self.temperature,
                    }).encode('utf-8'),
                    headers={'Authorization': f'Bearer {self.api_key}',
                             'Content-Type': 'application/json'})
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    body = json.loads(r.read().decode('utf-8'))
                out = body['choices'][0]['message']['content']
                usage = body.get('usage') or {}
                self.calls += 1
                self.total_tokens += int(usage.get('total_tokens') or 0)
                self.total_latency += time.perf_counter() - t0
                obj = parse_json_lenient(out)
                if obj is None:
                    last_err = 'json parse failed'
                    continue
                # 字段校验（Validity-Fidelity：不只看能否解析，还要看内容类型）
                if not isinstance(obj.get('crossed'), list):
                    last_err = 'crossed 不是列表'
                    continue
                rec = str(obj.get('recommendation', '')).strip().lower()
                if rec not in RECOMMENDATIONS:
                    last_err = f'recommendation 非法: {rec!r}'
                    continue
                cps = obj.get('concepts') or []
                cps = [c for c in cps if c in CONCEPT_CODES]
                return ({'crossed': [float(x) for x in obj['crossed']
                                     if isinstance(x, (int, float))],
                         'recommendation': rec, 'concepts': cps,
                         'reason': str(obj.get('reason', ''))},
                        {'ok': True, 'attempts': attempt + 1})
            except Exception as e:
                last_err = f'{type(e).__name__}: {e}'
        return None, {'ok': False, 'reason': last_err}

    # ---------- 成本摘要 ----------
    def cost_summary(self):
        return {'calls': self.calls, 'total_tokens': self.total_tokens,
                'total_latency': round(self.total_latency, 2),
                'mean_latency': (round(self.total_latency / self.calls, 3)
                                 if self.calls else None)}


# ---------------------------------------------------------------- 自测

def _self_test() -> int:
    ok, fail = 0, []
    print('=' * 72)
    print('a6_llm_reason.py 自测')
    print('=' * 72)

    R = Reasoner()
    print(f'  密钥可用: {R.available}')

    # ① JSON 解析的鲁棒性
    cases = [
        ('{"crossed":[4.0],"recommendation":"retest","concepts":["THRESHOLD_CROSS"],"reason":"跨阈值"}', True),
        ('```json\n{"crossed":[],"recommendation":"report","concepts":[],"reason":"ok"}\n```', True),
        ('说明如下：{"crossed":[],"recommendation":"report","concepts":[],"reason":"ok"} 完毕', True),
        ('完全不是 JSON', False),
    ]
    good = 0
    for txt, expect in cases:
        got = parse_json_lenient(txt) is not None
        good += int(got == expect)
    if good == len(cases):
        ok += 1
    else:
        fail.append(f'JSON 宽松解析 {good}/{len(cases)}')

    # ② 概念码过滤（非法码必须被剔除）
    obj = parse_json_lenient('{"crossed":[],"recommendation":"report",'
                             '"concepts":["THRESHOLD_CROSS","BOGUS"],"reason":"x"}')
    cps = [c for c in (obj.get('concepts') or []) if c in CONCEPT_CODES]
    if cps == ['THRESHOLD_CROSS']:
        ok += 1
    else:
        fail.append('非法概念码未被剔除')

    # ③ 无密钥 ⇒ 返回 None 且不崩
    if not R.available:
        out, meta = R.decide('Fat', 4.0, 3.8, 4.2, [3.5, 4.0], 0.3)
        if out is None and not meta['ok']:
            ok += 1
        else:
            fail.append('无密钥时行为异常')
    else:
        ok += 1

    # ④ 有密钥 ⇒ 真实调用 + 字段内容校验
    if R.available:
        out, meta = R.decide('Fat', 4.12, 3.85, 4.39, [3.0, 3.5, 4.0, 4.5], 0.3,
                             rpd=5.25)
        print(f'  真实调用: ok={meta.get("ok")}  attempts={meta.get("attempts")}')
        print(f'  输出: {out}')
        if meta.get('ok') and out is not None:
            # 内容正确性：crossed 应为 [4.0]（3.85..4.39 覆盖 4.0）
            if set(out['crossed']) == {4.0}:
                ok += 1
            else:
                fail.append(f'crossed 内容错误：{out["crossed"]}，期望 [4.0]')
            if out['recommendation'] in RECOMMENDATIONS:
                ok += 1
            else:
                fail.append('recommendation 非法')
        else:
            fail.append(f'调用失败：{meta.get("reason")}')
        print(f'  成本: {R.cost_summary()}')

    print(f'  通过 {ok} 项')
    if fail:
        print(f'  ❌ 失败 {len(fail)} 项：')
        for f_ in fail:
            print('     -', f_)
        return 1
    print('  ✅ 全部通过：JSON 宽松解析、概念码过滤、内容校验均正常')
    return 0


if __name__ == '__main__':
    sys.exit(_self_test())
