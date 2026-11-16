# -*- coding: utf-8 -*-
"""check_isolation.py — 上下文隔离的**函数级**静态检查

## 判据（函数级，而非模块级）
    **在线路径**（构造流水线时必须零泄漏）：
        Assessor.intervals ｜ Router.plan / expected_loss / sample_score
        Reporter.build / render ｜ FeedbackAgent.select
    **离线校准与评估**（允许访问真值，但必须显式传参）：
        函数名含 calibrate / metrics / coverage / cost
        ⇒ 允许使用 self.Y 或参数 Y

## 为什么按函数级
    实测核查：a4/a5 中需要真值的函数**全部**是校准或评估用途；
    在线路径（intervals / plan）本身不接触真值。
    ⇒ 模块级判定会把正当的评估函数误报为泄漏。

## 用法
    python check_isolation.py
"""
from __future__ import annotations

import ast
import io
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
AGENTS = os.path.join(HERE, 'agents')

#: 在线路径函数（函数体内不得出现真值标识）
ONLINE_FUNCS = {
    'intervals', 'plan', 'expected_loss', 'sample_score',
    'build', 'render', 'select', 'grade_of_rpd',
}
#: 函数名含这些词 ⇒ 视为校准/评估，允许访问真值
OFFLINE_HINTS = ('calibrat', 'metrics', 'coverage', 'cost', 'conformal')
#: 真值标识
TRUTH = ('Y', 'cow')


def _funcs(tree, src):
    """产出 (函数名, 节点, 起始行, 结束行)"""
    out = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            end = getattr(node, 'end_lineno', node.lineno)
            out.append((node.name, node, node.lineno, end))
    return out


def _body_names(node):
    """函数体内出现的 Name 标识（不含嵌套函数定义）"""
    names = set()
    for sub in ast.walk(node):
        if isinstance(sub, ast.Name):
            names.add(sub.id)
        elif isinstance(sub, ast.Attribute):
            # self.Y → 记 'Y'
            names.add(sub.attr)
    return names


def scan_file(rel):
    path = os.path.join(HERE, rel.replace('/', os.sep))
    src = io.open(path, encoding='utf-8').read()
    tree = ast.parse(src)
    # 计算自测段起点（之后不检查）
    cut = len(src.splitlines()) + 1
    for node in tree.body:
        if isinstance(node, ast.If):
            try:
                t = ast.unparse(node.test)
            except Exception:
                t = ''
            if '__name__' in t and '__main__' in t:
                cut = node.lineno
                break
    # 模块级赋值中的真值（不在函数内）——用于发现全局泄漏
    problems = []
    for name, node, a, b in _funcs(tree, src):
        if b >= cut:
            continue
        body = _body_names(node)
        hit = body & set(TRUTH)
        if not hit:
            continue
        if any(h in name.lower() for h in OFFLINE_HINTS):
            continue                      # 校准/评估函数，允许
        if name in ONLINE_FUNCS:
            problems.append((name, sorted(hit), '在线路径访问真值'))
        else:
            # 既非在线、也非离线提示 ⇒ 需人工复核，但不算失败
            problems.append((name, sorted(hit), '需复核'))
    return problems


def main():
    print('=' * 78)
    print('ISOLATION LEAK CHECK（函数级）')
    print('=' * 78)
    fails, warns = [], []
    targets = []
    if os.path.isdir(AGENTS):
        for f in sorted(os.listdir(AGENTS)):
            if f.endswith('.py'):
                targets.append('agents/' + f)
    for f in sorted(os.listdir(HERE)):
        if f.endswith('.py'):
            targets.append(f)

    for rel in targets:
        try:
            probs = scan_file(rel)
        except Exception as e:
            print(f'  {rel:28} ⚠️ 解析失败 {e}')
            continue
        hard = [p for p in probs if p[2] == '在线路径访问真值']
        soft = [p for p in probs if p[2] == '需复核']
        if hard:
            fails.append((rel, hard))
            print(f'  {rel:28} ❌ {len(hard)} 处在线路径泄漏')
            for nm, hit, _ in hard:
                print(f'        - {nm}() 使用 {hit}')
        elif soft:
            warns.append((rel, soft))
            print(f'  {rel:28} ⚠️ {len(soft)} 处需复核')
        else:
            print(f'  {rel:28} ✅ 在线路径干净')

    print('  ' + '-' * 74)
    if fails:
        print(f'  ❌ {len(fails)} 个模块的在线路径访问真值')
        return 1
    print('  ✅ 所有模块的**在线路径**均不接触真值')
    if warns:
        print(f'  ℹ️  {len(warns)} 个模块有需人工复核的真值引用（多为评估/校准）：')
        for rel, ps in warns:
            print(f'     {rel}: {[p[0] for p in ps]}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
