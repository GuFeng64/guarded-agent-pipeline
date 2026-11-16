# -*- coding: utf-8 -*-
"""run_tests.py — 一键运行 Agent 系统的全部自测

## 为什么需要
    之前每个模块的自测要逐个手工运行，容易漏。
    本脚本统一执行并汇总，作为"改完必须跑一遍"的闸门（章程规则 D2）。

## 用法
    python run_tests.py            # 全部
    python run_tests.py --fast     # 跳过需要真实 LLM 调用的
"""
from __future__ import annotations

import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))

#: (路径, 名称, 是否需要网络/密钥)
SUITES = [
    ('context.py', '上下文隔离', False),
    ('orchestrator.py', 'DAG + 并行 + 超时', False),
    ('degradation.py', '三档降级', False),
    ('assembly.py', '并行装配', False),
    ('agents/a2_branch.py', '分支 Agent', False),
    ('agents/a3_aggregate.py', '融合 + σ + TS 闭合', False),
    ('agents/a4_assess.py', '保形区间 + 分级', False),
    ('agents/a5_route.py', '损失无关路由', False),
    ('agents/a5_guarded.py', '受约束的 Agent', False),
    ('agents/a5_deliberation.py', '多 Agent 审议', False),
    ('agents/a6_report.py', '报告模板', False),
    ('agents/a6_llm.py', 'LLM 措辞层（无密钥也测回退）', False),
    ('agents/a6_llm_reason.py', 'LLM 推理层', False),
    ('agents/a7_feedback.py', '回流选样', False),
    ('agents/a1_acquisition.py', '采集闸门 + 尖峰修复', False),
    ('validator.py', '报告校验', False),
    ('pipeline.py', '统一流水线', False),
    ('process_isolation.py', '进程级隔离', False),
    ('isolated_pipeline.py', '隔离版流水线', False),
    ('check_isolation.py', '函数级泄漏检查', False),
    ('smoke_evaluate.py', '评估脚本冒烟测试', False),
    ('test_faults.py', '故障注入矩阵', False),
    ('test_audit_negctl.py', '证据阴性对照审计', False),
    ('test_negctl_suite.py', '检测器双向阴性对照', False),
    ('test_number_provenance.py', '文档数字溯源核对', False),
    ('test_crosscheck_v39.py', '与 v39 模型侧交叉核对', False),
    ('llm_ci.py', '置信区间计算', False),
    ('reliability_metrics.py', '可靠性指标（解析值自检）', False),
]


def run_one(rel, name):
    path = os.path.join(HERE, rel)
    t0 = time.perf_counter()
    p = subprocess.run([sys.executable, '-X', 'utf8', '-B', path],
                       capture_output=True, text=True, encoding='utf-8',
                       errors='replace', cwd=HERE)
    el = time.perf_counter() - t0
    out = (p.stdout or '') + (p.stderr or '')
    # 抽出"通过 N 项"
    n_pass = None
    for line in out.splitlines():
        if '通过' in line and '项' in line:
            try:
                n_pass = int(line.split('通过')[1].split('项')[0].strip())
            except Exception:
                pass
    return {'rel': rel, 'name': name, 'rc': p.returncode,
            'n': n_pass, 'elapsed': el, 'tail': out.strip().splitlines()[-2:]}


def main():
    fast = '--fast' in sys.argv
    print('=' * 78)
    print('AGENT SYSTEM — SELF-TEST SUITE')
    print('=' * 78)
    print(f'  {"module":26}{"items":>7}{"time":>9}  status')
    total_items, total_time, fails = 0, 0.0, []
    for rel, name, _needs in SUITES:
        r = run_one(rel, name)
        total_time += r['elapsed']
        if r['n']:
            total_items += r['n']
        st = '✅' if r['rc'] == 0 else '❌'
        if r['rc'] != 0:
            fails.append(r)
        print(f'  {rel:26}{(r["n"] if r["n"] is not None else "?"):>7}'
              f'{r["elapsed"]:>8.1f}s  {st}  {name}')
    print('  ' + '-' * 72)
    print(f'  {"TOTAL":26}{total_items:>7}{total_time:>8.1f}s  '
          f'{"✅ 全部通过" if not fails else f"❌ {len(fails)} 个失败"}')
    if fails:
        print()
        for r in fails:
            print(f'  ── 失败：{r["rel"]}（rc={r["rc"]}）')
            for line in r['tail']:
                print('     ' + line)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
