# -*- coding: utf-8 -*-
"""step527_guarded_value.py — Guarded Agent 在**争议样本**上是否比规则更有价值

## 问题
    step 前一次评估：regret 里 report=q·σ≈0.02，retest=1.0
    ⇒ "report" 永远明显最优 ⇒ 规则已是给定信息下的最优 ⇒ LLM 无增益（Δ=0）

    **但这只是因为成本尺度错了**：真实复测成本（约 0.15 个目标单位）
    远低于"报错值"的代价（|y−ŷ| 量级）。
    正确校准后，`q·σ` 会与复测成本相当 ⇒ 出现**争议样本**。

## 本脚本
    ① 用**真实成本尺度**重建 regret：
         report    : q·σ（该目标的典型误差）
         retest    : c_test（复测成本，按目标量纲设定）
         reference : 5·c_test（转参考法的固定代价）
    ② 找出**争议样本**（report 与 retest 的期望损失差 < 20%）
    ③ 在争议样本上比较：
         · rule       —— 规则决策
         · llm+guard  —— LLM 提议 + 门控
       用**真实损失**（基于 Y 的实际误差）评价，而不是代理 regret
    ④ 若 LLM 在争议样本上优于规则 ⇒ Agent 有决策价值

## 真实损失的构造（让两种动作可比）
    report 的错误代价 = |y − ŷ|
    retest 的代价     = c_test（复测后视为正确，不再有错误代价）
    reference 的代价  = c_ref（转参考法，视为正确但更贵）
"""
from __future__ import annotations

import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_AG = os.path.join(_HERE, 'agents')
_REV = os.path.dirname(_HERE)
for _p in (_HERE, _AG, os.path.join(_REV, 'code')):
    if _p not in sys.path:
        sys.path.insert(0, _p)

LOG = os.path.join(_REV, 'logs', 'step527_guarded_value.txt')
open(LOG, 'w', encoding='utf-8').close()


def log(*a):
    msg = ' '.join(str(x) for x in a)
    print(msg, flush=True)
    with open(LOG, 'a', encoding='utf-8') as fh:
        fh.write(msg + '\n')


def main(n_samples=20):
    from sklearn.model_selection import GroupKFold
    import mems_data as M
    from agents.a5_guarded import GuardedAgent, Guard

    d = M.load()
    Y = np.asarray(d['Y'], dtype=float)
    cow = np.asarray(d['cow'])
    T = list(d['targets'])
    n, nt = Y.shape
    folds = list(GroupKFold(5).split(np.zeros((n, 1)), Y[:, 0], cow))
    Ti = np.concatenate([te for _, te in folds])
    Z = np.load(os.path.join(_REV, 'experiments', 'agent_evaluate.npz'),
                allow_pickle=True)
    yhat, SIG, LO, HI, Q = Z['yhat'], Z['sigma'], Z['lo'], Z['hi'], Z['q']
    REP = ['Fat', 'Prot', 'TS']

    # ---- 成本尺度（按目标量纲）----
    C_TEST = {}
    for t in T:
        ti = T.index(t)
        C_TEST[t] = float(np.std(Y[Ti, ti]) * 0.03)   # 复测成本 ≈ 3% 的标准差

    grades = {}
    for t in T:
        ti = T.index(t)
        e = yhat[Ti, ti] - Y[Ti, ti]
        rmse = float(np.sqrt(np.mean(e ** 2)))
        rpd = float(Y[Ti, ti].std() / (rmse + 1e-12))
        r2 = float(1 - np.sum(e ** 2)
                   / np.sum((Y[Ti, ti] - Y[Ti, ti].mean()) ** 2))
        code = 'report' if rpd > 3 else ('good' if rpd > 2.5
                                        else ('screening' if rpd > 2 else 'reject'))
        grades[t] = {'r2': r2, 'rpd': rpd, 'code': code,
                     'reportable': code in ('report', 'good')}

    # ---- 代理 regret（LLM 可见）与真实损失（仅评估用）----
    def proxy_regret(i):
        out = {}
        for t in T:
            ix = T.index(t)
            if not grades[t]['reportable']:
                out[t] = {'report': 999.0, 'retest': C_TEST[t],
                          'reference': 5 * C_TEST[t]}
                continue
            q_t = float(Q[ix]) if np.isfinite(Q[ix]) else 0.30
            s = float(SIG[i, ix])
            out[t] = {'report': q_t * s,
                      'retest': C_TEST[t],
                      'reference': 5 * C_TEST[t]}
        return out

    def true_loss(i, t, action):
        ix = T.index(t)
        if action == 'report':
            return float(abs(Y[i, ix] - yhat[i, ix]))
        if action == 'retest':
            return C_TEST[t]
        return 5 * C_TEST[t]

    # ---- 找争议样本：某主目标的 report/retest 期望损失相近 ----
    rng = np.random.default_rng(0)
    contested = []
    for i in range(n):
        pr = proxy_regret(i)
        for t in REP:
            a, b = pr[t]['report'], pr[t]['retest']
            if max(a, b) > 0 and abs(a - b) / max(a, b) < 0.20:
                contested.append((i, t))
                break
    log('=' * 90)
    log('GUARDED AGENT ON CONTESTED CASES')
    log('=' * 90)
    log(f'  复测成本 C_TEST = {({t: round(C_TEST[t], 4) for t in REP})}')
    log(f'  争议样本数 = {len(contested)} / {n}')

    if not contested:
        log('  ⚠️ 没有争议样本 ⇒ 本协议无法区分 LLM 与规则（成本尺度仍不合适）')
        return 0

    pick = [c for c in contested][:n_samples]
    pick_i = sorted({i for i, _ in pick})
    log(f'  取前 {len(pick_i)} 个样本做评估')

    ag = GuardedAgent()
    log(f'  LLM 可用: {ag.available}')
    rows = []
    for k, i in enumerate(pick_i):
        pr = proxy_regret(i)
        ctx = {
            'targets': T,
            'lo': np.array([LO[i, T.index(t)] for t in T]),
            'hi': np.array([HI[i, T.index(t)] for t in T]),
            'sigma': np.array([SIG[i, T.index(t)] for t in T]),
            'q': np.array([Q[T.index(t)] if np.isfinite(Q[T.index(t)]) else 0.3
                           for t in T]),
            'grades': {t: grades[t] for t in T},
            'grade_thr': {'Fat': (4.0, 4.5, 5.0), 'Prot': (3.4, 3.7, 4.0),
                          'TS': (13.5, 14.5, 15.5), 'Lact': (), 'Urea': ()},
            'compliance': {'Fat': 3.1, 'Prot': 2.9, 'TS': None,
                           'Lact': None, 'Urea': None},
            'regret': pr, 'degraded': False, 'missing': [],
        }
        prop, gr, info = ag.decide(ctx)
        rule_acts = info.get('rule_actions', {})
        # 真实损失
        l_rule = sum(true_loss(i, t, rule_acts.get(t, 'reference')) for t in T)
        l_final = sum(true_loss(i, t, prop.actions.get(t, 'reference')) for t in T)
        l_opt = sum(min(true_loss(i, t, a) for a in ('report', 'retest', 'reference'))
                    for t in T)
        rows.append({'i': i, 'src': prop.source, 'ntool': prop.n_tool_calls,
                     'guard': info.get('guard_ok'),
                     'l_rule': l_rule, 'l_final': l_final, 'l_opt': l_opt})
        log(f'  [{k + 1}/{len(pick_i)}] i={i} src={prop.source:5} '
            f'tools={prop.n_tool_calls:2} guard={info.get("guard_ok")} '
            f'true_loss: rule={l_rule:.4f} final={l_final:.4f} opt={l_opt:.4f}')

    # ---- 汇总 ----
    log('')
    log('=' * 90)
    d_l = np.array([r['l_final'] - r['l_rule'] for r in rows])
    n_adopt = sum(1 for r in rows if r['src'] == 'llm')
    n_diff = int((np.abs(d_l) > 1e-9).sum())
    log(f'  采纳率 {n_adopt}/{len(rows)} ｜ 与规则不同的样本数 {n_diff}')
    log(f'  真实损失 Δ = {d_l.mean():+.5f}（负=LLM 更好）'
        f'  ｜ 更优 {int((d_l < -1e-9).sum())} 更差 {int((d_l > 1e-9).sum())}')
    regret_gap = np.array([r['l_rule'] - r['l_opt'] for r in rows])
    log(f'  规则的后悔值（相对最优）= {regret_gap.mean():+.5f}'
        f' ⇒ 规则本身离最优还有多少空间')
    log('')
    log('  判读：')
    log('    · 若 n_diff=0 ⇒ LLM 在争议样本上也与规则一致（无增益）')
    log('    · 若 Δ<0 且 n_diff>0 ⇒ **Agent 有决策价值**')
    log('    · 若 Δ>0 且 n_diff>0 ⇒ 门控需要收紧')
    log('=' * 90)


if __name__ == '__main__':
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 20)
