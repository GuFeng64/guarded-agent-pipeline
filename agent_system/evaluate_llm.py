# -*- coding: utf-8 -*-
"""evaluate_llm.py — LLM 推理层的评估（按 IEEE 2026 忠实性框架重做）

## 文献依据
    ① Right Answer, Wrong Reason (IEEE 2026)：
       正确率 ≠ 忠实性。需 ESI（重复一致性）/ CFS（概念消融）/ PSS（改写稳健性）
    ② Validity-Fidelity Gap (2026)：JSON 可解析 ≠ 内容正确
       ⇒ 逐字段核对内容，不把"能解析"当成功
    ③ ChemBench (ACL 2026 Findings)：分析化学场景基准

## 指标
    ① 准确率      crossed 与程序计算一致；recommendation 与参考规则一致
    ② ESI         同一输入重复 3 次，recommendation 一致率
    ③ CFS         对 LLM **引用的每个概念**做消融，看 recommendation 是否翻转
                  ⇒ 翻转率高 = 概念真的驱动决策；低 = 事后合理化
    ④ PSS         改写提问措辞，recommendation 一致率
    ⑤ 契约        字段类型正确 + 概念码合法（不只"能解析"）
    ⑥ 成本        延迟 / tokens / 调用数

## 参考规则（gold，确定性）
    crossed 非空            → retest
    区间半宽 > 2×容许半宽    → retest
    RPD < 2                 → reference
    否则                     → report

## 用法
    set LLM_API_KEY=... && python evaluate_llm.py [n_samples]
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

LOG = os.path.join(_REV, 'logs', 'agent_llm_eval2.txt')


def gold_rule(crossed, halfwidth, tolerance, rpd):
    if crossed:
        return 'retest'
    if halfwidth > 2 * tolerance:
        return 'retest'
    if rpd is not None and rpd < 2:
        return 'reference'
    return 'report'


def crossed_of(lo, hi, thr):
    return [float(t) for t in thr if lo <= t <= hi]


def ablate(concept, kw):
    """把一个概念对应的"触发条件"中性化，返回新 kwargs。"""
    k = dict(kw)
    if concept == 'THRESHOLD_CROSS':
        # 把区间平移到不含任何阈值的位置（保持宽度）
        w = k['hi'] - k['lo']
        mid = (k['lo'] + k['hi']) / 2
        thr = k['thresholds']
        cand = np.linspace(min(thr) - 3 * w - 1, max(thr) + 3 * w + 1, 400)
        for c in cand:
            if not any(c - w / 2 <= t <= c + w / 2 for t in thr):
                k['lo'], k['hi'] = float(c - w / 2), float(c + w / 2)
                break
    elif concept == 'WIDE_INTERVAL':
        k['tolerance'] = (k['hi'] - k['lo']) / 2 * 1.5 + 1e-6
    elif concept == 'HIGH_SIGMA':
        k['rpd'] = 10.0
    elif concept == 'DEGRADED':
        k['degraded'] = False
    return k


def main(n_samples=12):
    from agents.a6_llm_reason import Reasoner
    import mems_data as M

    fh = open(LOG, 'w', encoding='utf-8')

    def log(*a):
        msg = ' '.join(str(x) for x in a)
        print(msg, flush=True)
        fh.write(msg + '\n')

    log('=' * 88)
    log('LLM REASONING LAYER — EVALUATION (faithfulness-aware)')
    log('=' * 88)

    R = Reasoner()
    log(f'  密钥可用: {R.available}')
    if not R.available:
        log('  ⚠️ 无密钥 ⇒ 无法评估推理层；请设置 LLM_API_KEY')
        fh.close()
        return

    # ---- 数据：用 v39 的预测与区间 ----
    d = M.load()
    Y = np.asarray(d['Y'], dtype=float)
    T = list(d['targets'])
    n = len(Y)
    Z = np.load(os.path.join(_REV, 'experiments', 'agent_evaluate.npz'),
                allow_pickle=True)
    yhat, LO, HI = Z['yhat'], Z['lo'], Z['hi']
    from sklearn.model_selection import GroupKFold
    cow = np.asarray(d['cow'])
    folds = list(GroupKFold(5).split(np.zeros((n, 1)), Y[:, 0], cow))
    Ti = np.concatenate([te for _, te in folds])
    rpd_t = {}
    for t in T:
        ti = T.index(t)
        rmse = float(np.sqrt(np.mean((yhat[Ti, ti] - Y[Ti, ti]) ** 2)))
        rpd_t[t] = float(Y[Ti, ti].std() / (rmse + 1e-12))

    THR = {'Fat': [3.0, 3.5, 4.0, 4.5], 'Prot': [3.0, 3.2, 3.4],
           'TS': [11.5, 12.5, 13.5]}
    TOL = {'Fat': 0.30, 'Prot': 0.10, 'TS': 0.35}
    TARGETS = [t for t in ('Fat', 'Prot', 'TS')]

    rng = np.random.default_rng(0)
    idxs = np.sort(rng.choice(n, size=n_samples, replace=False))
    log(f'  样本数 {n_samples} ｜ 目标 {TARGETS}')

    # ---------- 逐样本评估 ----------
    acc_cross, acc_rec = [], []
    esi, pss = [], []
    cfs_num, cfs_den = 0, 0
    contract_ok = 0
    contract_tot = 0
    cfs_detail = []

    for k, i in enumerate(idxs):
        for t in TARGETS:
            ti = T.index(t)
            kw = dict(yhat=float(yhat[i, ti]), lo=float(LO[i, ti]),
                      hi=float(HI[i, ti]), thresholds=THR[t],
                      tolerance=TOL[t], rpd=rpd_t[t], degraded=False)
            base, meta = R.decide(t, **kw)
            contract_tot += 1
            if base is None:
                continue
            contract_ok += 1

            # ① 准确率
            true_cross = crossed_of(kw['lo'], kw['hi'], kw['thresholds'])
            acc_cross.append(set(np.round(base['crossed'], 6))
                             == set(np.round(true_cross, 6)))
            gold = gold_rule(true_cross, (kw['hi'] - kw['lo']) / 2,
                             kw['tolerance'], kw['rpd'])
            acc_rec.append(base['recommendation'] == gold)

            # ② ESI：重复 3 次
            reps = [base['recommendation']]
            for _ in range(2):
                o2, _m2 = R.decide(t, **kw)
                if o2 is not None:
                    reps.append(o2['recommendation'])
            esi.append(len(set(reps)) == 1)

            # ③ CFS：对每个引用概念做消融
            for c in base['concepts']:
                kw2 = ablate(c, kw)
                o3, _m3 = R.decide(t, **kw2)
                cfs_den += 1
                if o3 is not None and o3['recommendation'] != base['recommendation']:
                    cfs_num += 1
                    cfs_detail.append((t, c, base['recommendation'],
                                       o3['recommendation']))

            # ④ PSS：改写措辞
            o4, _m4 = R.decide(t, paraphrase=True, **kw)
            if o4 is not None:
                pss.append(o4['recommendation'] == base['recommendation'])

        if (k + 1) % 4 == 0:
            log(f'    ...已处理 {k + 1}/{n_samples} 个样本，'
                f'累计 {R.calls} 次调用')

    # ---------- 汇总 ----------
    log('')
    log('  ── 结果 ──')
    log(f'  ① 阈值跨越判断准确率   {np.mean(acc_cross):.1%}  (n={len(acc_cross)})')
    log(f'     建议与参考规则一致率 {np.mean(acc_rec):.1%}  (n={len(acc_rec)})')
    log(f'  ② ESI 重复一致性       {np.mean(esi):.1%}  (n={len(esi)})')
    log(f'  ③ CFS 概念消融翻转率   {cfs_num}/{cfs_den} = '
        f'{(cfs_num / cfs_den if cfs_den else float("nan")):.1%}')
    log(f'     （翻转 = 该概念真的驱动了决策；高才好）')
    log(f'  ④ PSS 改写一致性       '
        f'{np.mean(pss) if pss else float("nan"):.1%}  (n={len(pss)})')
    log(f'  ⑤ 契约遵守（字段合法） {contract_ok}/{contract_tot} = '
        f'{contract_ok / contract_tot:.1%}')
    cs = R.cost_summary()
    log(f'  ⑥ 成本：调用 {cs["calls"]} 次 ｜ tokens {cs["total_tokens"]} ｜ '
        f'平均延迟 {cs["mean_latency"]}s')
    if cfs_detail:
        log('     CFS 翻转示例（前 5 条）：')
        for t, c, a, b in cfs_detail[:5]:
            log(f'       {t}: 消融 {c} ⇒ 建议 {a} → {b}')

    fh.close()
    print('\n  saved:', LOG)


if __name__ == '__main__':
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 12
    main(n)
