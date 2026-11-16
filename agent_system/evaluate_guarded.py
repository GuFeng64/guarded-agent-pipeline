# -*- coding: utf-8 -*-
"""evaluate_guarded.py — Guarded Agent 的**真实评估**

## 核心问题（必须答）
    门控后的 LLM 决策，比**纯规则**更好还是更差？
      · 若门控总拒绝 ⇒ 回退 ⇒ 与规则完全相同（无害无益）
      · 若门控采纳且 LLM 更差 ⇒ **有害**
      · 若门控采纳且 LLM 更好 ⇒ 有价值

## 度量
    ① 工具使用：LLM 平均调用几次工具（证明是真 Agent 循环）
    ② 门控通过率（按规则统计拒绝原因）
    ③ 与规则决策的一致率
    ④ **决策质量**：用期望损失评价
          · rule_only   : 规则决策的期望损失
          · guarded_llm : 门控后最终决策的期望损失
        Δ = guarded_llm − rule_only（负值 = 更好）
    ⑤ 配对 bootstrap（按牛重抽）检验 Δ 的显著性

## 说明
    期望损失来自 A5 路由模型（用 σ 与 q 积分的代理），**不需要真值**。
    因此本评估可在无真值情况下进行；真值只用于最终的对照报告，不参与决策。
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_AG = os.path.join(_HERE, 'agents')
_REV = os.path.dirname(_HERE)
for _p in (_HERE, _AG, os.path.join(_REV, 'code')):
    if _p not in sys.path:
        sys.path.insert(0, _p)

LOG = os.path.join(_REV, 'logs', 'evaluate_guarded.txt')
open(LOG, 'w', encoding='utf-8').close()


def log(*a):
    msg = ' '.join(str(x) for x in a)
    print(msg, flush=True)
    with open(LOG, 'a', encoding='utf-8') as fh:
        fh.write(msg + '\n')


def build_ctx(i, targets, yhat, SIG, LO, HI, Q, grades, regret_all,
              degraded=False, missing=None):
    ix = {t: targets.index(t) for t in targets}
    return {
        'targets': list(targets),
        'lo': np.array([LO[i, ix[t]] for t in targets]),
        'hi': np.array([HI[i, ix[t]] for t in targets]),
        'sigma': np.array([SIG[i, ix[t]] for t in targets]),
        'q': np.array([Q[ix[t]] for t in targets]),
        'grades': {t: grades[t] for t in targets},
        'grade_thr': {'Fat': (4.0, 4.5, 5.0), 'Prot': (3.4, 3.7, 4.0),
                      'TS': (13.5, 14.5, 15.5), 'Lact': (), 'Urea': ()},
        'compliance': {'Fat': 3.1, 'Prot': 2.9, 'TS': None,
                       'Lact': None, 'Urea': None},
        'regret': {t: regret_all[i][t] for t in targets if t in regret_all[i]},
        'degraded': degraded,
        'missing': list(missing or []),
    }


def main(n_samples=12):
    from sklearn.model_selection import GroupKFold
    import mems_data as M
    from agents.a5_guarded import GuardedAgent, Guard, ACTION_CODES

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

    # 分级
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

    # 期望损失（来自 A5 的损失无关路由，**不用真值**）
    from agents.a5_route import Router, LOSSES
    R = Router(T, reportable=REP)
    regret_all = {}
    for i in range(n):
        ri = {}
        for t in T:
            if not grades[t]['reportable']:
                ri[t] = {'report': 99.0, 'retest': 1.0, 'reference': 0.5}
                continue
            ix = T.index(t)
            q_t = float(Q[ix]); s = float(SIG[i, ix])
            # 代理期望损失：retest 的成本 = c_test；report 的成本 ∝ q·σ；
            # reference 的成本取定值（转参考法的固定代价）
            ri[t] = {'report': q_t * s,
                     'retest': 1.0,
                     'reference': 3.0}
        regret_all[i] = ri

    rng = np.random.default_rng(0)
    idxs = np.sort(rng.choice(n, size=min(n_samples, n), replace=False))

    ag = GuardedAgent()
    log('=' * 92)
    log('GUARDED AGENT EVALUATION')
    log('=' * 92)
    log(f'  LLM 可用: {ag.available} ｜ 样本数 {len(idxs)}')
    if not ag.available:
        log('  ⚠️ 无 LLM_API_KEY ⇒ 只能验证回退路径')

    rows = []
    for k, i in enumerate(idxs):
        ctx = build_ctx(i, T, yhat, SIG, LO, HI, Q, grades, regret_all)
        prop, gr, info = ag.decide(ctx)
        # 计算两种决策的期望损失
        def total_loss(actions):
            s = 0.0
            for t, a in actions.items():
                s += float(regret_all[i].get(t, {}).get(a, 99.0))
            return s
        l_rule = total_loss(info.get('rule_actions', {}))
        l_final = total_loss(prop.actions)
        rows.append({
            'i': int(i), 'source': prop.source,
            'n_tool': int(prop.n_tool_calls),
            'guard_ok': info.get('guard_ok'),
            'violations': info.get('violations') or [],
            'l_rule': l_rule, 'l_final': l_final,
            'llm_used': info.get('llm_used', False),
            'error': info.get('error'),
        })
        log(f'  [{k + 1}/{len(idxs)}] i={i} source={prop.source:5} '
            f'tools={prop.n_tool_calls} guard_ok={info.get("guard_ok")} '
            f'loss rule={l_rule:.3f} final={l_final:.3f}')

    # ---------- 汇总 ----------
    log('')
    log('=' * 92)
    log('  汇总')
    log('=' * 92)
    n_used = sum(1 for r in rows if r['llm_used'])
    n_adopt = sum(1 for r in rows if r['source'] == 'llm')
    n_tools = [r['n_tool'] for r in rows if r['llm_used']]
    log(f'  ① LLM 实际可用：{n_used}/{len(rows)}')
    log(f'  ② 门控采纳（最终用 LLM 决策）：{n_adopt}/{len(rows)} '
        f'= {n_adopt / max(len(rows), 1):.1%}')
    if n_tools:
        log(f'  ③ 工具调用：平均 {np.mean(n_tools):.2f} 次/样本'
            f' ｜ 范围 {min(n_tools)}–{max(n_tools)}')

    viol = {}
    for r in rows:
        for v in r['violations']:
            key = v.split(' ')[0]
            viol[key] = viol.get(key, 0) + 1
    if viol:
        log(f'  ④ 门控拒绝原因：{viol}')

    d_loss = np.array([r['l_final'] - r['l_rule'] for r in rows])
    log(f'  ⑤ 决策质量：Δ期望损失 = {d_loss.mean():+.4f}'
        f'（负值=优于规则）  标准差 {d_loss.std():.4f}')
    n_better = int((d_loss < -1e-9).sum())
    n_worse = int((d_loss > 1e-9).sum())
    log(f'     更优 {n_better} ｜ 更差 {n_worse} ｜ 相同 {len(d_loss) - n_better - n_worse}')

    # 配对 bootstrap（按样本重抽）
    if len(d_loss) >= 5:
        boots = []
        for _ in range(2000):
            s = rng.choice(d_loss, size=len(d_loss), replace=True)
            boots.append(s.mean())
        boots = np.asarray(boots)
        lo, hi = np.percentile(boots, [2.5, 97.5])
        p = float(np.mean(boots < 0))
        log(f'  ⑥ Bootstrap：95% CI [{lo:+.4f}, {hi:+.4f}] ｜ '
            f'P(优于规则) = {p:.3f}')

    log('')
    log('=' * 92)
    log('  判读：')
    log('    · 若采纳率低且 Δ≈0 ⇒ 门控把 LLM 挡在门外，安全但无收益')
    log('    · 若采纳率高且 Δ<0 ⇒ LLM 提议确有价值（配合门控）')
    log('    · 若采纳率高但 Δ>0 ⇒ **门控不够严，需收紧**')
    log('=' * 92)

    np.savez(os.path.join(_REV, 'experiments', 'evaluate_guarded.npz'),
             idx=np.array([r['i'] for r in rows]),
             source=np.array([r['source'] for r in rows]),
             n_tool=np.array([r['n_tool'] for r in rows]),
             l_rule=np.array([r['l_rule'] for r in rows]),
             l_final=np.array([r['l_final'] for r in rows]))
    log('')
    log('  saved')


if __name__ == '__main__':
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 12)
