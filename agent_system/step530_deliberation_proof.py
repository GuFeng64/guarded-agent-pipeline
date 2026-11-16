# -*- coding: utf-8 -*-
"""step530_deliberation_proof.py — 证明审议环在**该触发时确实触发**

## 为什么需要
    step529 实测：8 个正常样本中
      · A1 被调用 2/8 ✅（agent-as-tool 真实发生）
      · 修订 0/8     ← LLM 首轮就合规，审议环从未启动

    ⇒ 审议环是"安全网"，但**未证明它能工作**。
    ⇒ 必须造出让门控**必然触发**的场景。

## 触发场景
    S1 **整族缺失退化**：degraded=True, missing=['b7','b8','b9']
       LLM 很可能仍说 report ⇒ G3 触发 ⇒ 期望它修订为 reference/retest
    S2 **不可报告目标**：把 Lact 的可报告性设为 True 但要求 report
       （更直接：让 LLM 面对"Lact 看起来很准"的假象）
    S3 **G6 反向期望损失**：让 report 的期望损失最高
       ⇒ 若 LLM 仍说 report，G6 触发

## 判据
    任一场景下出现 rounds ≥ 2 且 revisions ≥ 1 ⇒ 审议环被证明可工作
"""
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_AG = os.path.join(_HERE, 'agents')
_REV = os.path.dirname(_HERE)
for _p in (_HERE, _AG, os.path.join(_REV, 'code')):
    if _p not in sys.path:
        sys.path.insert(0, _p)

LOG = os.path.join(_REV, 'logs', 'step530_deliberation.txt')
open(LOG, 'w', encoding='utf-8').close()


def log(*a):
    msg = ' '.join(str(x) for x in a)
    print(msg, flush=True)
    with open(LOG, 'a', encoding='utf-8') as fh:
        fh.write(msg + '\n')


def base_ctx():
    return {
        'targets': ['Fat', 'Prot', 'Lact', 'TS', 'Urea'],
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


def main(trials=3):
    import mems_data as M
    from agents.a5_deliberation import MultiAgent

    d = M.load()
    RV = ('Multi SE14', 'Multi_new SE20', 'Reflection SE20', 'Multi SE25')
    views = {v: d['X'][v][0].astype(np.float64) for v in RV}

    ma = MultiAgent()
    log('=' * 88)
    log('DELIBERATION LOOP — PROOF THAT IT CAN FIRE')
    log('=' * 88)
    log(f'  LLM 可用: {ma.available}')

    SCEN = {}
    # S1 整族缺失退化：LLM 很可能仍说 report ⇒ G3 应触发
    c1 = base_ctx()
    c1['degraded'] = True
    c1['missing'] = ['b7', 'b8', 'b9']
    SCEN['S1 整族缺失去退化'] = c1

    # S2 G6：让 report 的期望损失最高
    c2 = base_ctx()
    for t in ('Fat', 'Prot', 'TS'):
        c2['regret'][t] = {'report': 9.0, 'retest': 1.0, 'reference': 2.0}
    SCEN['S2 report 期望损失最高'] = c2

    # S3 组合（同时退化 + G6）
    c3 = base_ctx()
    c3['degraded'] = True
    c3['missing'] = ['b7', 'b8', 'b9']
    for t in ('Fat', 'Prot', 'TS'):
        c3['regret'][t] = {'report': 9.0, 'retest': 1.0, 'reference': 2.0}
    SCEN['S3 退化 + G6'] = c3

    fired = False
    for name, ctx in SCEN.items():
        log('')
        log(f'  ── {name} ──')
        for k in range(trials):
            prop, crit, info = ma.decide(ctx, views, verbose=False)
            tr = info.get('transcript', [])
            log(f'     第 {k + 1} 次：rounds={info["rounds"]} '
                f'revisions={info["revisions"]} '
                f'accepted={info.get("accepted")} fallback={info["fallback"]} '
                f'A1={info["a1_calls"]}')
            for t in tr:
                log(f'        round {t["round"]}: ok={t["critique_ok"]} '
                    f'{t["violations"][:2]}')
            if info['rounds'] >= 2 and info['revisions'] >= 1:
                fired = True

    log('')
    log('=' * 88)
    if fired:
        log('  ✅ **审议环被证明可工作**：出现 rounds≥2 且 revisions≥1')
        log('     ⇒ 审计 Agent 的质疑确实改变了提议方的决策')
    else:
        log('  ⚠️ 三个刻意构造的场景**都未触发**审议环')
        log('     ⇒ 说明 LLM 在退化与反向损失提示下也给出了合规决策')
        log('     ⇒ 审议环仍只是"单元测试验证过的安全网"，未在真实 LLM 下被触发')
    log('=' * 88)
    return 0 if fired else 0


if __name__ == '__main__':
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 3)
