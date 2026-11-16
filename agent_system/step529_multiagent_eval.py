# -*- coding: utf-8 -*-
"""step529_multiagent_eval.py — 多 Agent 的**真实 LLM 评估**

## 测什么
    ① A5b 是否**主动调用 A1**（agent-as-tool 是否真发生）
    ② V 的质疑是否**触发修订**（审议环是否真运转）
    ③ 轮数分布（1 = 一次通过；>1 = 经过修订）
    ④ 最终是否都通过门控（或回退）
    ⑤ 与规则决策的一致率
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

LOG = os.path.join(_REV, 'logs', 'step529_multiagent.txt')
open(LOG, 'w', encoding='utf-8').close()


def log(*a):
    msg = ' '.join(str(x) for x in a)
    print(msg, flush=True)
    with open(LOG, 'a', encoding='utf-8') as fh:
        fh.write(msg + '\n')


def main(n_samples=10):
    from sklearn.model_selection import GroupKFold
    import mems_data as M
    from agents.a5_deliberation import MultiAgent

    d = M.load()
    Y = np.asarray(d['Y'], dtype=float)
    cow = np.asarray(d['cow'])
    T = list(d['targets'])
    n, nt = Y.shape
    RV = ('Multi SE14', 'Multi_new SE20', 'Reflection SE20', 'Multi SE25')
    folds = list(GroupKFold(5).split(np.zeros((n, 1)), Y[:, 0], cow))
    Ti = np.concatenate([te for _, te in folds])
    Z = np.load(os.path.join(_REV, 'experiments', 'agent_evaluate.npz'),
                allow_pickle=True)
    yhat, SIG, LO, HI, Q = Z['yhat'], Z['sigma'], Z['lo'], Z['hi'], Z['q']
    REP = ['Fat', 'Prot', 'TS']

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

    rng = np.random.default_rng(0)
    idxs = np.sort(rng.choice(n, size=min(n_samples, n), replace=False))
    ma = MultiAgent()
    log('=' * 92)
    log('MULTI-AGENT DELIBERATION — REAL LLM EVALUATION')
    log('=' * 92)
    log(f'  LLM 可用: {ma.available} ｜ 样本数 {len(idxs)}')

    rows = []
    for k, i in enumerate(idxs):
        ctx = {
            'targets': T,
            'lo': np.array([LO[i, T.index(t)] for t in T]),
            'hi': np.array([HI[i, T.index(t)] for t in T]),
            'sigma': np.array([SIG[i, T.index(t)] for t in T]),
            'q': np.array([Q[T.index(t)] for t in T]),
            'grades': {t: grades[t] for t in T},
            'grade_thr': {'Fat': (4.0, 4.5, 5.0), 'Prot': (3.4, 3.7, 4.0),
                          'TS': (13.5, 14.5, 15.5), 'Lact': (), 'Urea': ()},
            'compliance': {'Fat': 3.1, 'Prot': 2.9, 'TS': None,
                           'Lact': None, 'Urea': None},
            'regret': {t: {'report': float(Q[T.index(t)]),
                           'retest': 1.0, 'reference': 3.0}
                       for t in REP},
            'degraded': False, 'missing': [],
        }
        views = {v: d['X'][v][i].astype(np.float64) for v in RV}
        prop, crit, info = ma.decide(ctx, views, verbose=False)
        rows.append({'i': int(i), 'rounds': info['rounds'],
                     'a1': info['a1_calls'], 'rev': info['revisions'],
                     'fb': info['fallback'], 'src': prop.source,
                     'accepted': info.get('accepted', False),
                     'tools': info['n_tool_calls']})
        log(f"  [{k + 1}/{len(idxs)}] i={i} src={prop.source:5} "
            f"rounds={info['rounds']} A1调用={info['a1_calls']} "
            f"修订={info['revisions']} 工具={info['n_tool_calls']} "
            f"接受={info.get('accepted')} 回退={info['fallback']}")

    log('')
    log('=' * 92)
    log('  汇总')
    log('=' * 92)
    n_a1 = sum(1 for r in rows if r['a1'] > 0)
    n_rev = sum(1 for r in rows if r['rev'] > 0)
    n_acc = sum(1 for r in rows if r['accepted'])
    n_fb = sum(1 for r in rows if r['fb'])
    log(f'  ① 调用过 A1：{n_a1}/{len(rows)}  '
        f'（{"✅ agent-as-tool 真实发生" if n_a1 else "⚠️ 未触发"}）')
    log(f'  ② 发生过修订：{n_rev}/{len(rows)}  '
        f'（{"✅ 审议环真实运转" if n_rev else "（首轮即通过）"}）')
    log(f'  ③ 最终通过门控：{n_acc}/{len(rows)} ｜ 回退规则：{n_fb}')
    log(f'  ④ 平均工具调用 {np.mean([r["tools"] for r in rows]):.1f} 次')
    log(f'  ⑤ 轮数分布：{ {r: sum(1 for x in rows if x["rounds"] == r) for r in sorted(set(x["rounds"] for x in rows))} }')
    log('')
    log('  判读：')
    log('    · A1 被调用 ⇒ 决策 Agent 真的**跨 Agent 请求了感知能力**')
    log('    · 修订 > 0 ⇒ 审计 Agent 的质疑**真的改变了决策**')
    log('    · 二者同时成立 ⇒ 这是**真多 Agent 交互**，不是单 Agent 包装')
    log('=' * 92)


if __name__ == '__main__':
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 10)
