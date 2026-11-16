# -*- coding: utf-8 -*-
"""demo/run_demo.py — **决策层端到端演示**（合成数据，无需原始数据/模型）

## 演示什么
    本仓库是**决策层**：输入上游模型给出的 ŷ 与 σ，输出可审计的处置决策。
    本脚本用**合成数据**模拟上游模型，然后把决策层完整跑一遍：

        A4 评估：保形区间 + RPD 分级（覆盖率是否达标）
        A5 路由：三档预算下的复测规划（损失函数无关）
        A5b/A5c：受约束的 Agent（工具查询 + 8 条门控）
                  多 Agent 审议（提议-质疑-修订）
        A6 报告：结构化文本（数值可溯源）
        V  校验：5 项合规检查 + 文本数值一致性
        A7 回流：标记高价值样本
        故障演示：分支缺失的降级判定

## 为什么不用真实模型
    模型（深度学习/TabPFN）属于上游，不在本仓库范围内。
    决策层对模型是**不可知的** —— 它只需要 ŷ 与 σ。

## 用法
    python demo/run_demo.py
"""
from __future__ import annotations

import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
_AS = os.path.join(_REPO, 'agent_system')
for _p in (_AS, os.path.join(_AS, 'agents'), os.path.join(_REPO, 'code')):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from sklearn.linear_model import Ridge                       # noqa: E402
from sklearn.model_selection import GroupKFold                # noqa: E402

import mems_data as M                                         # noqa: E402
import metrics as MT                                          # noqa: E402


def rule(title):
    print()
    print('=' * 78)
    print(title)
    print('=' * 78)


def fit_upstream(d):
    """**模拟上游模型**：每视图一个岭回归，用 GroupKFold 得到无泄漏的 OOF 预测。

    σ 用"各视图预测的分歧度"近似（与真实系统的融合思想一致）。
    """
    Y = np.asarray(d['Y'], dtype=float)
    cow = np.asarray(d['cow'])
    T = list(d['targets'])
    n, nt = Y.shape
    views = M.VIEWS
    folds = list(GroupKFold(5).split(np.zeros((n, 1)), Y[:, 0], cow))
    per_view = {v: np.zeros((n, nt)) for v in views}
    yhat = np.zeros((n, nt))
    for fi, (tr, te) in enumerate(folds):
        for v in views:
            X = d['X'][v].astype(np.float64)
            mu = X[tr].mean(0)
            sd = X[tr].std(0) + 1e-9
            Xtr = (X[tr] - mu) / sd
            Xte = (X[te] - mu) / sd
            # SNV
            Xtr = (Xtr - Xtr.mean(1, keepdims=True)) / (Xtr.std(1, keepdims=True) + 1e-9)
            Xte = (Xte - Xte.mean(1, keepdims=True)) / (Xte.std(1, keepdims=True) + 1e-9)
            for ti in range(nt):
                m = Ridge(alpha=10.0).fit(Xtr, Y[tr, ti])
                per_view[v][te, ti] = m.predict(Xte)
        # 等权融合
        yhat[te] = np.mean([per_view[v][te] for v in views], axis=0)
    # σ：视图间分歧（作为可靠性信号的代理）
    stack = np.stack([per_view[v] for v in views], axis=0)   # (V, n, nt)
    sigma = stack.std(0)
    return Y, cow, T, folds, yhat, sigma, per_view


def main():
    print('=' * 78)
    print('GUARDED AGENT PIPELINE — DECISION LAYER DEMO')
    print('=' * 78)
    print('  数据：合成（结构同真实研究数据，数值不可用于科研结论）')
    print('  模型：不含（本仓库是上游模型的**下游决策层**）')

    d = M.load()
    Y, cow, T, folds, yhat, sigma, per_view = fit_upstream(d)
    n, nt = Y.shape
    Ti = np.concatenate([te for _, te in folds])
    # ⚠️ 真实系统中「业务上报口径」与「统计分级」恰好一致（Lact/Urea 的 RPD
    #    只有 1.59/1.07 ⇒ 自动判 reject）。合成数据上 Lact 偏容易（RPD≈2.7）
    #    ⇒ 二者会冲突。为保证 demo 自洽，此处**从实际分级推导**上报口径。
    REPORTABLE = [t for t in T if gre] if False else None   # 占位，稍后赋值

    rule('① 上游模型精度（本仓库不负责，仅作为输入）')
    r2 = MT.r2_by_target(Y, yhat, Ti)
    gap = float(sum(0.98 - r2[t] for t in MT.MAIN3))
    for t in T:
        ti = T.index(t)
        e = yhat[Ti, ti] - Y[Ti, ti]
        rmse = float(np.sqrt(np.mean(e ** 2)))
        rpd = float(Y[Ti, ti].std() / (rmse + 1e-12))
        print(f'  {t:6} R²={r2[t]:+.4f}  RMSE={rmse:.4f}  RPD={rpd:5.2f}')
    print(f'  主目标 gap = {gap:.4f}（越低越好）')

    # ---------------- A4 评估 ----------------
    rule('② A4 评估：保形区间 + RPD 分级')
    from agents.a4_assess import Assessor
    A4 = Assessor(folds, Y, T)
    ev = A4.run(yhat, alpha=0.10)
    Q = ev['q']
    LO, HI = ev['lo'], ev['hi']
    grades = ev['grades']
    REPORTABLE = [t for t in T if grades[t]['reportable']]
    cov3 = float(np.mean([ev['coverage'][t] for t in REPORTABLE]))
    print(f'  名义 90% ⇒ 上报目标实测覆盖率 {cov3:.1%}')
    for t in T:
        g = ev['grades'][t]
        print(f'  {t:6} RPD={g["rpd"]:5.2f}  grade={g["code"]:10} '
              f'reportable={g["reportable"]}')
    print(f'  上报口径（由分级推导）：{REPORTABLE}')
    print('  ⚠️ 真实系统中该口径为业务设定 [Fat, Prot, TS]，且与分级一致')

    # ---------------- A5 路由 ----------------
    rule('③ A5 路由：损失函数无关的复测规划')
    from agents.a5_route import Router, GRADE_THRESHOLDS, COMPLIANCE_LIMITS
    R = Router(T, reportable=REPORTABLE)
    for loss in ('linear', 'grade', 'compliance'):
        pr = R.plan(yhat, sigma, Q, 0.20, policy='expected_loss', loss=loss)
        print(f'  loss={loss:11} budget={pr["budget"]:3}  '
              f'retest={len(pr["retest_idx"]):3}')
    print(f'  分档阈值 {GRADE_THRESHOLDS}')
    print(f'  合规限值 { {k: v for k, v in COMPLIANCE_LIMITS.items() if v} }')

    # ---------------- A5b 受约束的 Agent ----------------
    rule('④ A5b Guarded Agent（工具查询 + 8 条门控）')
    from agents.a5_guarded import GuardedAgent, Guard

    def mk_ctx(i):
        return {
            'targets': T,
            'lo': np.array([LO[i, T.index(t)] for t in T]),
            'hi': np.array([HI[i, T.index(t)] for t in T]),
            'sigma': np.array([sigma[i, T.index(t)] for t in T]),
            'q': np.array([Q[T.index(t)] for t in T]),
            'grades': {t: grades[t] for t in T},
            'grade_thr': {**GRADE_THRESHOLDS, 'Lact': (), 'Urea': ()},
            'compliance': {**COMPLIANCE_LIMITS, 'Lact': None, 'Urea': None},
            'regret': {t: {'report': float(Q[T.index(t)]),
                           'retest': 1.0, 'reference': 3.0}
                       for t in REPORTABLE},
            'degraded': False, 'missing': [],
        }

    ag = GuardedAgent()
    print(f'  LLM 可用: {ag.available}（无密钥 ⇒ 纯规则回退，这是设计行为）')
    ctxs = [mk_ctx(i) for i in range(n)]
    props, diags = ag.decide_batch(ctxs, budget_frac=0.20, verbose=True)
    n_rule = sum(1 for p in props if p.source == 'rule')
    print(f'  决策来源：规则 {n_rule} ｜ LLM {n - n_rule}')
    g0 = Guard().check(props[0], ctxs[0])
    print(f'  首样本门控结果：ok={g0.ok}  checks={len(g0.checked)} 条')

    # ---------------- A5c 多 Agent 审议 ----------------
    rule('⑤ A5c 多 Agent 审议（提议 → 质疑 → 修订）')
    from agents.a5_deliberation import MultiAgent, CriticAgent, SpectrumTools
    spec = SpectrumTools({v: d['X'][v][0].astype(np.float64) for v in M.VIEWS})
    a1 = spec.call('request_spectral_recheck', {})
    print(f'  agent-as-tool：A1 复查 ⇒ verdict={a1["verdict"]} '
          f'corrected={a1["corrected"]} spike={a1["spike_index"]:.4f}')
    bad = props[0].__class__(
        actions={**props[0].actions, 'Lact': 'report'},
        reasons=dict(props[0].reasons))
    crit = CriticAgent().review(bad, ctxs[0])
    print(f'  质疑一个违规提议 ⇒ ok={crit.ok}')
    for h in crit.hints:
        print(f'    建议：{h}')
    ma = MultiAgent()
    print(f'  MultiAgent LLM 可用: {ma.available}（无密钥 ⇒ 纯规则）')

    # ---------------- A6 报告 + V 校验 ----------------
    rule('⑥ A6 报告 + V 校验（唯一可见真值的环节）')
    from agents.a6_report import Reporter
    from validator import Validator
    Rep = Reporter(T)
    Val = Validator(Y, cow, T, reportable=REPORTABLE)
    plan = R.plan(yhat, sigma, Q, 0.20, policy='expected_loss', loss='grade')
    n_ok = 0
    for i in range(n):
        r = Rep.build(sample_id=f'S{i:03d}', yhat_row=yhat[i],
                      lo_row=LO[i], hi_row=HI[i],
                      decision=plan['decision'][i], grades=grades)
        txt = Rep.render(r)
        vr = Val.validate(r, txt, i, yhat[i])
        n_ok += int(vr.ok)
    print(f'  校验通过率 {n_ok}/{n} = {n_ok / n:.1%}')
    print('  ── 第 0 号样本的报告 ──')
    r0 = Rep.build(sample_id='S000', yhat_row=yhat[0], lo_row=LO[0],
                   hi_row=HI[0], decision=plan['decision'][0], grades=grades)
    for line in Rep.render(r0).split('\n'):
        print('  ' + line)

    # ---------------- A7 回流 ----------------
    rule('⑦ A7 回流：标记高价值样本')
    from agents.a7_feedback import FeedbackAgent
    fb = FeedbackAgent().select(plan['decision'], plan['score'], list(range(n)))
    print(f'  高价值样本 {len(fb.high_value_ids)} 个（供人工复检后回流训练）')

    # ---------------- 故障演示 ----------------
    rule('⑧ 故障演示：分支缺失的降级判定')
    from degradation import decide as deg_decide
    for miss in ([], ['b9'], ['b8'], ['b7', 'b8', 'b9']):
        dd = deg_decide(miss)
        print(f'  缺失 {str(miss):20} ⇒ action={dd.action:16} '
              f'Δgap={dd.delta:+.4f}')

    # ---------------- 汇总 ----------------
    rule('汇总')
    print(f'  上游模型 gap        {gap:.4f}')
    print(f'  保形覆盖率          {cov3:.1%}（名义 90%）')
    print(f'  校验通过率          {n_ok / n:.1%}')
    print(f'  高价值回流样本      {len(fb.high_value_ids)}')
    print()
    print('  ⚠️ 合成数据 ⇒ 上述数值**不可用于科研结论**，仅用于演示接口与流程。')
    print('     原始数据集上的实测值见仓库 README 的「实测结果」一节。')


if __name__ == '__main__':
    main()
