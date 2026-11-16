# -*- coding: utf-8 -*-
"""evaluate.py — 端到端评估（基于 pipeline.py，v2）

## 相对 v1 的变化
    v1 直接调用各 Agent 且使用**旧的路由 API**（R.plan(yhat, LO, HI, ...)），
    在 a5_route 升级为"损失函数无关"后已失效。
    v2 改为通过 `NIRPipeline` 串联，并用新的损失无关规划器。

## 产出
    ① 模型重建：gap 是否复现 0.0324（含 TS 闭合）
    ② 覆盖率：名义 90% / 95%
    ③ 逐目标指标与 RPD 分级
    ④ 复测节省率（按 σ 拒识）
    ⑤ **三种损失结构**下的规划对比（linear / grade / compliance）
    ⑥ 降级准确性（与退化表比对）
    ⑦ 校验通过率
    ⑧ A1 采集闸门的整批通过率
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_AG = os.path.join(_HERE, 'agents')
_REV = os.path.dirname(_HERE)
for _p in (_HERE, _AG, os.path.join(_REV, 'code')):
    if _p not in sys.path:
        sys.path.insert(0, _p)

LOG = os.path.join(_REV, 'logs', 'agent_evaluate.txt')
OUT = os.path.join(_REV, 'experiments', 'agent_evaluate.npz')


def main():
    from sklearn.model_selection import GroupKFold
    import mems_data as M
    import metrics as MT
    from orchestrator import Orchestrator, CANONICAL_BRANCHES
    from degradation import DEGRADATION_TABLE
    from agents.a2_branch import BranchAgent
    from agents.a3_aggregate import Aggregator
    from agents.a4_assess import Assessor
    from agents.a5_route import Router, GRADE_THRESHOLDS, COMPLIANCE_LIMITS
    from pipeline import NIRPipeline, PipelineConfig

    fh = open(LOG, 'w', encoding='utf-8')

    def log(*a):
        msg = ' '.join(str(x) for x in a)
        print(msg, flush=True)
        fh.write(msg + '\n')

    d = M.load()
    Y = np.asarray(d['Y'], dtype=float)
    cow = np.asarray(d['cow'])
    T = list(d['targets'])
    n, nt = Y.shape
    folds = list(GroupKFold(5).split(np.zeros((n, 1)), Y[:, 0], cow))
    Ti = np.concatenate([te for _, te in folds])
    B = np.load(os.path.join(_REV, 'experiments',
                             'step426_4v_vi4_cache.npz'), allow_pickle=True)
    MU = [B[f'f{fi}_mu'] for fi in range(5)]
    SD = [B[f'f{fi}_sd'] for fi in range(5)]
    REPORTABLE = ('Fat', 'Prot', 'TS')

    log('=' * 88)
    log('END-TO-END EVALUATION (pipeline v2)  n=%d nt=%d' % (n, nt))
    log('=' * 88)

    # ---------------- ① 分支并行推理 ----------------
    t0 = time.perf_counter()

    def call(nm):
        ag = BranchAgent(nm, folds)
        return {'tr': [ag.fold_train(fi) for fi in range(5)],
                'te': [ag.fold_test(fi) for fi in range(5)]}

    orch = Orchestrator(timeout=30.0, max_workers=5)
    res = orch.run_branches(list(CANONICAL_BRANCHES), call)
    _, degraded, missing = orch.assemble(res)
    t_par = time.perf_counter() - t0
    preds = {nm: res[nm].pred for nm in CANONICAL_BRANCHES if res[nm].ok}

    t0 = time.perf_counter()
    for nm in CANONICAL_BRANCHES:
        call(nm)
    t_ser = time.perf_counter() - t0
    log('')
    log(f'  ① 分支推理：并行 {t_par:.2f}s ｜ 串行 {t_ser:.2f}s'
        f'（缓存回放，验证编排而非推理加速）')
    log(f'     degraded={degraded} missing={missing}')

    # ---------------- ② 融合 + TS 闭合 ----------------
    A3 = Aggregator(folds, Y, MU, SD, T, cow=cow)
    agg = A3.run(preds, apply_ts_closure=True)
    yhat, sigma, alphas = agg['yhat'], agg['sigma'], agg['alpha']
    r2_all = MT.r2_by_target(Y, yhat, Ti)
    gap = float(sum(0.98 - r2_all[t] for t in MT.MAIN3))
    r2_base = MT.r2_by_target(Y, agg['base'], Ti)
    gap_base = float(sum(0.98 - r2_base[t] for t in MT.MAIN3))
    log('')
    log(f'  ② 融合：闭合前 gap {gap_base:.4f} ｜ 闭合后 **{gap:.4f}**'
        f'（v39 定稿 0.0324）')
    log(f'     alpha = {np.round(alphas, 2).tolist()}')

    # ---------------- ③ 区间与分级 ----------------
    A4 = Assessor(folds, Y, T)
    for a in (0.10, 0.05):
        ev = A4.run(yhat, alpha=a)
        cov3 = float(np.mean([ev['coverage'][t] for t in REPORTABLE]))
        log(f'     名义 {1 - a:.0%} ⇒ 三主目标实测覆盖率 {cov3:.1%}')
    ev90 = A4.run(yhat, alpha=0.10)
    grades = ev90['grades']
    LO, HI, Q = ev90['lo'], ev90['hi'], ev90['q']
    log('')
    log('  ③ 逐目标指标与分级：')
    log(f'     {"target":7}{"R²":>9}{"RMSE":>10}{"RPD":>8}  grade')
    for t in T:
        g = grades[t]
        log(f'     {t:7}{g["r2"]:>+9.4f}{g["rmse"]:>10.4f}{g["rpd"]:>8.2f}'
            f'  {g["code"]}')

    # ---------------- ④ 复测节省率 ----------------
    log('')
    log('  ④ 复测节省率（按 σ 拒识）：')
    log(f'     {"budget":>8}{"report":>8}{"R2_avg":>10}{"RPD_avg":>10}')
    for bf in (0.0, 0.10, 0.20, 0.30, 0.40):
        keep = np.ones((n, nt), dtype=bool)
        if bf > 0:
            for fi, (tr, te) in enumerate(folds):
                for ti in range(nt):
                    tau = np.quantile(sigma[tr, ti], 1 - bf)
                    keep[te, ti] = sigma[te, ti] <= tau
        r2s, rpds = [], []
        for t in REPORTABLE:
            ti = T.index(t)
            m = keep[Ti, ti]
            yk, pk = Y[Ti, ti][m], yhat[Ti, ti][m]
            r2 = 1.0 - np.sum((yk - pk) ** 2) / (np.sum((yk - yk.mean()) ** 2) + 1e-12)
            rmse = float(np.sqrt(np.mean((yk - pk) ** 2)))
            r2s.append(r2)
            rpds.append(float(yk.std() / (rmse + 1e-12)))
        log(f'     {bf:>7.0%}{1 - bf:>8.0%}{np.mean(r2s):>10.4f}'
            f'{np.mean(rpds):>10.2f}')

    # ---------------- ⑤ 三种损失结构 ----------------
    log('')
    log('  ⑤ 损失无关规划对比（σ 排序 vs 期望损失）：')
    log(f'     分级阈值（固定规划值）：{GRADE_THRESHOLDS}')
    log(f'     国标限值：{{k: v for k, v in COMPLIANCE_LIMITS.items() if v}}')
    R = Router(T)
    for loss in ('linear', 'grade', 'compliance'):
        log('')
        log(f'     ── loss={loss} ──')
        log(f'     {"target":7}{"budget":>8}{"greedy_σ":>12}{"plan":>12}'
            f'{"plan 更优":>11}  note')
        # ⚠️ 标注空检验：该损失结构下若无法触发，改善必然为 0
        n_trig = {}
        if loss == 'compliance':
            for t in REPORTABLE:
                lim = COMPLIANCE_LIMITS.get(t)
                n_trig[t] = (0 if lim is None
                             else int((Y[:, T.index(t)] < lim).sum()))
        elif loss == 'grade':
            for t in REPORTABLE:
                thr = GRADE_THRESHOLDS.get(t)
                n_trig[t] = (0 if thr is None else int(
                    np.sum(np.abs(np.searchsorted(thr, Y[:, T.index(t)])
                                  - np.searchsorted(thr, yhat[:, T.index(t)])) > 0)))
        else:
            for t in REPORTABLE:
                n_trig[t] = -1
        for t in REPORTABLE:
            for bf in (0.20, 0.30):
                c = {}
                for pol in ('greedy_sigma', 'expected_loss'):
                    pr = R.plan(yhat, sigma, Q, bf, policy=pol,
                                loss=loss, target=t)
                    c[pol] = R.cost_of(pr, yhat, Y, loss=loss, target=t)
                rel = (c['greedy_sigma'] - c['expected_loss']) / max(
                    c['greedy_sigma'], 1e-9) * 100
                note = ''
                if n_trig.get(t, -1) == 0:
                    note = ' ⚠️空检验（样本不触发该情形）'
                elif n_trig.get(t, -1) > 0:
                    note = f' 触发 {n_trig[t]} 例'
                log(f'     {t:7}{bf:>7.0%}{c["greedy_sigma"]:>12.1f}'
                    f'{c["expected_loss"]:>12.1f}{rel:>10.1f}%{note}')

    # ---------------- ⑥ 降级准确性 ----------------
    log('')
    log('  ⑥ 降级准确性（基准为**不含 TS 闭合**的 base gap）：')
    log(f'     基准 = {gap_base:.4f}（应 0.0329）')
    for nm in CANONICAL_BRANCHES:
        sub = {k: v for k, v in preds.items() if k != nm}
        a2 = A3.run(sub, apply_ts_closure=False)
        r2_2 = MT.r2_by_target(Y, a2['yhat'], Ti)
        g2 = float(sum(0.98 - r2_2[t] for t in MT.MAIN3))
        exp = DEGRADATION_TABLE.get(nm, float('nan'))
        tag = '✅' if abs((g2 - gap_base) - exp) < 1.5e-3 else '⚠️'
        log(f'     丢 {nm:4}: Δ={g2 - gap_base:+.4f}  表值 {exp:.4f}  {tag}')

    # ---------------- ⑦ 走 pipeline（含校验 + A7）----------------
    log('')
    log('  ⑦ 通过统一流水线（pipeline.py）：')
    pipe = NIRPipeline(PipelineConfig(reportable=list(REPORTABLE)),
                       folds=folds, Y=Y, cow=cow)
    out = pipe.run(yhat=yhat, sigma=sigma, q=Q, Y=Y, cow=cow,
                   budget_frac=0.20, loss='grade')
    log(f'     {out["log"][0] if out["log"] else "(no log)"}')
    log(f'     校验通过率 {out.get("validation_pass", float("nan")):.1%}')
    log(f'     高价值样本 {len(out["feedback"].high_value_ids)} 个')

    # ---------------- ⑧ A1 采集闸门 ----------------
    log('')
    log('  ⑧ A1 采集闸门（整批）：')
    from agents.a1_acquisition import AcquisitionAgent
    A1 = AcquisitionAgent()
    RV = ('Multi SE14', 'Multi_new SE20', 'Reflection SE20', 'Multi SE25')
    Xv = {v: d['X'][v].astype(np.float64) for v in RV}
    n_acc = n_cor = 0
    for i in range(n):
        r = A1.check_one({v: Xv[v][i] for v in RV})
        if r.verdict == 'ACCEPT':
            n_acc += 1
            n_cor += int(r.corrected)
    log(f'     ACCEPT {n_acc}/{n} = {n_acc / n:.1%} ｜ 其中修复后接受 {n_cor}')

    log('')
    log('=' * 88)
    log(f'  核心：gap {gap:.4f} ｜ 覆盖 {np.mean([ev90["coverage"][t] for t in REPORTABLE]):.1%}'
        f' ｜ 校验 {out.get("validation_pass", 0):.1%}')
    log('=' * 88)
    fh.close()

    np.savez(OUT, Y=Y, cow=cow, yhat=yhat, sigma=sigma, lo=LO, hi=HI, q=Q,
             alphas=alphas, gap=gap, gap_base=gap_base,
             coverage=np.array([ev90['coverage'][t] for t in T]))
    print('\n  saved:', LOG)


if __name__ == '__main__':
    main()
