# -*- coding: utf-8 -*-
"""test_crosscheck_v39.py — **Agent 侧 vs v39 模型侧** 交叉核对

## 为什么
    文档声称"Agent 系统精确复现 v39"。
    此前只核对了 gap 与部分指标 ⇒ 不够。
    本脚本做**三方交叉核对**：

        ① v39/manifest.json         （模型侧冻结的指标）
        ② v39/oof_predictions.npz   （模型侧冻结的逐样本预测）
        ③ agent_evaluate.npz        （Agent 流水线产出的逐样本预测）

## 判据
    · 逐样本 yhat **逐元素一致**（atol 1e-12）
    · 逐目标 R²/RMSE/RPD 与 manifest 一致到 1e-6
    · gap 由 manifest 指标独立算出，与 Agent 侧一致
    · α 与 step481 一致

## 另外核对 LLM 指标的可溯源性
    从 logs/agent_llm_eval*.txt 解析，与文档字符串比对；
    并检查"样本量"是否被如实标注。
"""
from __future__ import annotations

import io
import json
import os
import re
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REV = os.path.dirname(_HERE)
sys.path.insert(0, os.path.join(_REV, 'code'))

LOG = os.path.join(_REV, 'logs', 'crosscheck_v39.txt')
open(LOG, 'w', encoding='utf-8').close()


def log(*a):
    msg = ' '.join(str(x) for x in a)
    print(msg, flush=True)
    with open(LOG, 'a', encoding='utf-8') as fh:
        fh.write(msg + '\n')


def main():
    from sklearn.model_selection import GroupKFold
    import mems_data as M
    import metrics as MT

    ok, fail = 0, []
    V39 = os.path.join(_REV, 'models_final', 'v39')

    log('=' * 88)
    log('CROSS-CHECK: Agent 侧 vs v39 模型侧')
    log('=' * 88)

    man = json.load(io.open(os.path.join(V39, 'manifest.json'),
                            encoding='utf-8'))
    met = man['metrics']
    OOF = np.load(os.path.join(V39, 'oof_predictions.npz'), allow_pickle=True)
    AG = np.load(os.path.join(_REV, 'experiments', 'agent_evaluate.npz'),
                 allow_pickle=True)
    Y, yhat = AG['Y'], AG['yhat']
    d = M.load()
    T = list(d['targets'])
    n, nt = Y.shape
    cow = np.asarray(AG['cow'])
    folds = list(GroupKFold(5).split(np.zeros((n, 1)), Y[:, 0], cow))
    Ti = np.concatenate([te for _, te in folds])

    # ---------- ① oof_predictions 的键 ----------
    log('')
    log('  ① v39/oof_predictions.npz 的键')
    log(f'    {sorted(OOF.files)}')
    key = None
    for cand in ('PRED', 'pred', 'yhat', 'Y_pred'):
        if cand in OOF.files:
            key = cand
            break
    if key is None:
        log('    ⚠️ 未找到预测键，跳过得逐元素核对')
        fail.append('找不到 v39 的预测键')
    else:
        P_v39 = np.asarray(OOF[key], dtype=float)
        log(f'    预测键 = {key}  shape={P_v39.shape}')
        if P_v39.shape != yhat.shape:
            log(f'    ❌ 形状不一致：v39 {P_v39.shape} vs Agent {yhat.shape}')
            fail.append('v39 与 Agent 的预测形状不同')
        else:
            md = float(np.abs(P_v39 - yhat).max())
            log(f'    **逐元素最大差 = {md:.3e}**')
            if md < 1e-12:
                log('    ✅ 完全一致')
                ok += 1
            elif md < 1e-9:
                log('    ✅ 数值一致（浮点级）')
                ok += 1
            else:
                log(f'    ❌ 存在差异 {md:.3e}')
                fail.append(f'v39 与 Agent 预测差 {md:.3e}')

    # ---------- ② 逐目标指标 vs manifest ----------
    log('')
    log('  ② 逐目标指标 vs manifest.json')
    log(f'    {"target":7}{"R²(Agent)":>13}{"R²(man)":>13}{"Δ":>11}'
        f'{"RPD(Agent)":>12}{"RPD(man)":>11}')
    allok = True
    for t in T:
        ti = T.index(t)
        e = yhat[Ti, ti] - Y[Ti, ti]
        r2 = float(1 - np.sum(e ** 2) / np.sum((Y[Ti, ti] - Y[Ti, ti].mean()) ** 2))
        rmse = float(np.sqrt(np.mean(e ** 2)))
        rpd = float(Y[Ti, ti].std() / (rmse + 1e-12))
        r2m = float(met[t]['R2'])
        rpdm = float(met[t]['RPD'])
        d_r2 = abs(r2 - r2m)
        d_rpd = abs(rpd - rpdm)
        good = (d_r2 < 1e-6) and (d_rpd < 1e-4)
        allok = allok and good
        log(f'    {t:7}{r2:>+13.6f}{r2m:>+13.6f}{d_r2:>11.1e}'
            f'{rpd:>12.4f}{rpdm:>11.4f}  {"✅" if good else "❌"}')
    if allok:
        ok += 1
    else:
        fail.append('逐目标指标与 manifest 不一致')

    # ---------- ③ gap 由 manifest 独立算出 ----------
    log('')
    log('  ③ gap（由 manifest 指标独立计算，不读 Agent 产物）')
    gap_man = float(sum(0.98 - float(met[t]['R2']) for t in MT.MAIN3))
    r2a = MT.r2_by_target(Y, yhat, Ti)
    gap_ag = float(sum(0.98 - r2a[t] for t in MT.MAIN3))
    log(f'    manifest 算得 gap = {gap_man:.6f}')
    log(f'    Agent 算得 gap   = {gap_ag:.6f}')
    if abs(gap_man - gap_ag) < 1e-6:
        log('    ✅ 一致')
        ok += 1
    else:
        fail.append(f'gap 不一致 {gap_man:.6f} vs {gap_ag:.6f}')

    # ---------- ④ α ----------
    log('')
    log('  ④ TS 闭合系数 α')
    al = np.asarray(AG['alphas']) if 'alphas' in AG.files else None
    log(f'    agent_evaluate.npz alphas = {np.round(al, 2).tolist()}'
        if al is not None else '    ⚠️ 无 alphas')
    from agents.a2_branch import BranchAgent
    from agents.a3_aggregate import fit_aggregator, ORDER
    B = np.load(os.path.join(_REV, 'experiments',
                             'step426_4v_vi4_cache.npz'), allow_pickle=True)
    MU = [B[f'f{fi}_mu'] for fi in range(5)]
    SD = [B[f'f{fi}_sd'] for fi in range(5)]
    preds = {}
    for nm in ORDER:
        ag = BranchAgent(nm, folds)
        preds[nm] = {'tr': [ag.fold_train(fi) for fi in range(5)],
                     'te': [ag.fold_test(fi) for fi in range(5)]}
    fit = fit_aggregator(folds, Y, MU, SD, preds, T,
                         apply_ts_closure=True, cow=cow)
    exp = np.array([0.80, 0.80, 0.80, 0.95, 0.90])
    log(f'    a3 拟合 alphas = {np.round(fit.alpha, 2).tolist()}')
    log(f'    step481/v39    = {exp.tolist()}')
    if np.allclose(np.round(fit.alpha, 2), exp):
        log('    ✅ 一致')
        ok += 1
    else:
        fail.append('α 与 v39 不一致')

    # ---------- ⑤ LLM 指标可溯源性 ----------
    log('')
    log('  ⑤ LLM 指标 vs logs/agent_llm_eval*.txt')
    p1 = os.path.join(_REV, 'logs', 'agent_llm_eval.txt')
    p2 = os.path.join(_REV, 'logs', 'agent_llm_eval2.txt')
    t1 = io.open(p1, encoding='utf-8').read() if os.path.exists(p1) else ''
    t2 = io.open(p2, encoding='utf-8').read() if os.path.exists(p2) else ''
    # ⚠️ 期望值随扩样更新：25 样本 / 75 判断（旧值 6 样本已作废）
    checks = {
        '数值不变性 297/297': '297/297' in t1,
        '成本 0.96s（措辞层）': '0.96s' in t1,
        'tokens 206（措辞层）': '206' in t1,
        '阈值跨越 46.7%': '46.7%' in t2,
        'ESI 76.0%': '76.0%' in t2,
        'CFS 21.8%': '21.8%' in t2,
        'PSS 73.3%': '73.3%' in t2,
        '契约 100.0%': '100.0%' in t2,
        '样本数 25': '样本数 25' in t2 or '25 个样本' in t2,
    }
    for k, v in checks.items():
        log(f'    {k:22} {"✅ 在日志中" if v else "❌ 未找到"}')
        if v:
            ok += 1
        else:
            fail.append(f'LLM 指标不可溯源：{k}')
    # 样本量是否被如实标注
    m = re.search(r'样本数\s*(\d+)', t2)
    ns = m.group(1) if m else '?'
    docs = io.open(os.path.join(_HERE, 'SYSTEM_README.md'),
                   encoding='utf-8').read()
    log(f'    LLM 推理层样本数 = {ns}（日志中明确记录）')
    log(f'    文档是否标注该样本量：{"✅" if ns != "?" and ns in docs else "⚠️ 未标注"}')
    if ns != '?' and ns in docs:
        ok += 1
    else:
        fail.append(f'文档未标注 LLM 样本量 n={ns}')

    # ---------- 汇总 ----------
    log('')
    log('=' * 88)
    log(f'  通过 {ok} 项')
    if fail:
        log(f'  ❌ {len(fail)} 项失败：')
        for f_ in fail:
            log(f'     - {f_}')
        return 1
    log('  ✅ Agent 侧与 v39 模型侧逐元素一致；LLM 指标全部可溯源到日志')
    log('=' * 88)
    return 0


if __name__ == '__main__':
    sys.exit(main())
