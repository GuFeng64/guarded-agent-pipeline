# -*- coding: utf-8 -*-
"""smoke_evaluate.py — `evaluate.py` 的**冒烟测试**（纳入 run_tests.py）

## 为什么需要
    evaluate.py 曾因 a5_route 接口变更而**失效且无人发现**
    （18 个测试套件都没有覆盖它）。
    本冒烟测试只验证"能跑通且关键板块在"，不做完整数值校验（那太慢）。

## 判据
    ① 退出码为 0
    ② 日志中同时出现 8 个板块的标题
    ③ 关键数字（gap 0.0324）出现在日志里
"""
import io
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
LOG = os.path.join(os.path.dirname(HERE), 'logs', 'agent_evaluate.txt')
NEED = ['① 分支推理', '② 融合', '③ 逐目标', '④ 复测节省率',
        '⑤ 损失无关规划', '⑥ 降级准确性', '⑦ 通过统一流水线', '⑧ A1 采集闸门']


def main():
    p = subprocess.run([sys.executable, '-X', 'utf8', '-B',
                        os.path.join(HERE, 'evaluate.py')],
                       capture_output=True, text=True, encoding='utf-8',
                       errors='replace', cwd=HERE)
    ok, fail = 0, []
    if p.returncode == 0:
        ok += 1
    else:
        fail.append(f'evaluate.py 退出码 {p.returncode}: '
                    f'{(p.stderr or "")[-300:]}')
    txt = io.open(LOG, encoding='utf-8').read() if os.path.exists(LOG) else ''
    miss = [n for n in NEED if n not in txt]
    if not miss:
        ok += 1
    else:
        fail.append(f'日志缺少板块：{miss}')
    if re.search(r'闭合后 \*\*0\.0324\*\*|闭合后 gap = 0\.0324', txt):
        ok += 1
    else:
        fail.append('日志中未出现 gap 0.0324')

    print('=' * 72)
    print('smoke_evaluate.py 冒烟测试')
    print('=' * 72)
    print(f'  通过 {ok} 项')
    if fail:
        print(f'  ❌ 失败 {len(fail)} 项：')
        for f_ in fail:
            print('     -', f_)
        return 1
    print('  ✅ 全部通过：evaluate.py 可跑通，8 个板块齐全，gap 0.0324 出现')
    return 0


if __name__ == '__main__':
    sys.exit(main())
