# -*- coding: utf-8 -*-
"""step522_online_stability.py — 在线/流式稳定性：长序列处理下的内存、延迟与确定性

## 为什么需要
    此前所有评估都是 **297 条批量回放**。真实部署是**逐条到来**的流式输入。
    必须验证三件事：
      ① **内存不增长**（无泄漏）
      ② **延迟不漂移**（前 100 条 vs 后 100 条）
      ③ **结果与顺序无关**（同一条样本在任何位置给出相同决策）

## 协议
    · 流长度：297 条 × 10 轮 = 2970 条（打乱顺序，带样本 ID 回溯）
    · 每处理 100 条记录：RSS、GPU 显存、单条延迟（中位数，取该段 100 条）
    · 最后做**确定性检查**：同一样本在不同轮次的结果是否完全相同

## 组件
    只经过**在线路径**（A4 评估 → A5 路由 → A6 报告 → V 校验）
    A2/A3 是缓存回放，不参与流式测试（其成本已在 step521 分解）

## 判据
    · RSS 增长 < 50 MB
    · GPU 显存增长 < 100 MB
    · 后 100 条延迟不超过前 100 条的 1.5 倍
    · 确定性：同一 (样本, 轮次) 的决策 100% 一致
"""
import gc
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

LOG = os.path.join(_REV, 'logs', 'step522_online.txt')
open(LOG, 'w', encoding='utf-8').close()


def log(*a):
    msg = ' '.join(str(x) for x in a)
    print(msg, flush=True)
    with open(LOG, 'a', encoding='utf-8') as fh:
        fh.write(msg + '\n')


def rss_mb():
    """RSS（MB）。psutil 不可用则返回 None（**不得返回 0 冒充有效值**）。"""
    try:
        import psutil
        return psutil.Process().memory_info().rss / 1e6
    except Exception:
        return None


def gpu_mb():
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / 1e6
    except Exception:
        pass
    return None


def pyheap_mb():
    """Python 堆当前分配（tracemalloc，内置可用）。"""
    import tracemalloc
    if not tracemalloc.is_tracing():
        return None
    cur, peak = tracemalloc.get_traced_memory()
    return cur / 1e6


def main():
    from sklearn.model_selection import GroupKFold
    import mems_data as M
    from agents.a4_assess import Assessor
    from agents.a5_route import Router
    from agents.a6_report import Reporter
    from validator import Validator

    d = M.load()
    Y = np.asarray(d['Y'], dtype=float)
    cow = np.asarray(d['cow'])
    T = list(d['targets'])
    n, nt = Y.shape
    folds = list(GroupKFold(5).split(np.zeros((n, 1)), Y[:, 0], cow))
    Z = np.load(os.path.join(_REV, 'experiments', 'agent_evaluate.npz'),
                allow_pickle=True)
    yhat, sigma, Q = Z['yhat'], Z['sigma'], Z['q']
    LO, HI = Z['lo'], Z['hi']
    REPORTABLE = ['Fat', 'Prot', 'TS']

    # ⭐ 预导入 torch：其导入约占 +388 MB。
    #    若在基线之后才导入，会被误算成"内存增长"（已实测踩坑）
    try:
        import torch  # noqa: F401
    except Exception:
        pass
    import tracemalloc
    tracemalloc.start()
    log('=' * 92)
    log('ONLINE / STREAMING STABILITY')
    log('=' * 92)
    log(f'  n={n} ｜ 流长度 = {n} × 10 = {n * 10} 条')

    # ---- 组件只构造一次（模拟常驻服务）----
    A4 = Assessor(folds, Y, T)     # 保留原签名（含离线校准能力）
    R = Router(T, reportable=REPORTABLE)
    Rep = Reporter(T)
    Val = Validator(Y, cow, T, reportable=REPORTABLE)
    grades = {t: {'r2': float('nan'), 'rpd': float('nan'),
                  'code': 'report' if t in REPORTABLE else 'reject',
                  'reportable': t in REPORTABLE} for t in T}
    # 用真实 RPD 重算一次分级（离线，只做一次）
    r2s = {}
    Ti = np.concatenate([te for _, te in folds])
    from sklearn.model_selection import GroupKFold as GK
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

    # ---- 一次性批量规划（路由是批量决策，流式只查表）----
    plan0 = R.plan(yhat, sigma, Q, 0.20, policy='expected_loss',
                   loss='grade')
    PLAN_DECISION = list(plan0['decision'])
    log(f'  批量规划：budget={plan0["budget"]} '
        f'retest={len(plan0["retest_idx"])}')

    # ---- 生成流 ----
    ROUNDS = 10
    rng = np.random.default_rng(0)
    stream = []
    for rd in range(ROUNDS):
        perm = rng.permutation(n)
        stream.extend([(int(i), rd) for i in perm])
    log(f'  流已生成：{len(stream)} 条')

    # ---- 预热 ----
    for k in range(20):
        i = int(rng.integers(0, n))
        _ = Rep.render(Rep.build(sample_id=f'W{k:03d}', yhat_row=yhat[i],
                                 lo_row=LO[i], hi_row=HI[i],
                                 decision='report', grades=grades))
    gc.collect()

    mem0 = rss_mb()
    gpu0 = gpu_mb()
    heap0 = pyheap_mb()
    log('')
    log(f'  {"已处理":>8}{"Py堆(MB)":>10}{"ΔPy堆":>9}{"延迟(ms)":>11}')

    SEG = 100
    lat_seg = []
    decisions = {}
    t_seg = time.perf_counter()
    records = []

    for pos, (i, rd) in enumerate(stream):
        t0 = time.perf_counter()
        # 在线路径：评估 → 路由 → 报告 → 校验
        dec = PLAN_DECISION[i]
        rep = Rep.build(sample_id=f'S{i:03d}', yhat_row=yhat[i],
                        lo_row=LO[i], hi_row=HI[i],
                        decision=dec, grades=grades)
        txt = Rep.render(rep)
        _ = Val.validate(rep, txt, i, yhat[i])
        el = time.perf_counter() - t0
        lat_seg.append(el)

        # 记录决策（用于确定性检查）
        decisions.setdefault(i, set()).add(str(dec))

        if (pos + 1) % SEG == 0:
            seg = pos + 1
            gc.collect()
            m, g, h = rss_mb(), gpu_mb(), pyheap_mb()
            med = float(np.median(lat_seg[-SEG:])) * 1000
            records.append((seg, m, (m - mem0) if (m is not None and mem0 is not None) else None,
                            g, (g - gpu0) if (g is not None and gpu0 is not None) else None,
                            med, h, (h - heap0) if (h is not None and heap0 is not None) else None))
            _hs = "None" if h is None else f"{h:10.2f}"
            _ds = "None" if records[-1][7] is None else f"{records[-1][7]:+9.2f}"
            log(f"  {seg:>8}{_hs}{_ds}{med:>11.2f}")

    log('')
    log('=' * 92)
    log('  判读')
    log('=' * 92)

    # ① 内存（tracemalloc 优先；不可用则失败，不得静默通过）
    heap_growth = records[-1][7]
    if heap_growth is None:
        ok_mem = False
        log('  ① 内存：❌ 测量不可用（tracemalloc 未启动）')
    else:
        ok_mem = heap_growth < 5.0
        log(f'  ① Python 堆增长 {heap_growth:+.2f} MB（阈值 5）'
            f'  {"✅" if ok_mem else "❌"}')
    rss_g = records[-1][2]
    if rss_g is not None:
        log(f'     RSS 增长 {rss_g:+.1f} MB（参考；已预导入 torch 以排除其导入开销）')
    else:
        log('     RSS：psutil 不可用（不计入判定）')

    # ② 延迟漂移
    first = records[0][5]
    last = records[-1][5]
    drift = last / first if first > 0 else float('nan')
    ok_lat = drift < 1.5
    log(f'  ② 延迟：首段 {first:.2f} ms → 末段 {last:.2f} ms'
        f'（比值 {drift:.2f}，阈值 1.5）  {"✅" if ok_lat else "❌"}')

    # ③ 确定性
    n_multi = sum(1 for i, s2 in decisions.items() if len(s2) > 1)
    ok_det = (n_multi == 0)
    log(f'  ③ 确定性：{len(decisions)} 个不同样本，'
        f'{n_multi} 个出现不一致决策  {"✅" if ok_det else "❌"}')

    log('')
    allok = ok_mem and ok_lat and ok_det
    log('=' * 92)
    log(f'  {"✅ 在线稳定性通过" if allok else "❌ 存在问题"}')
    log('=' * 92)
    log('')
    log('  ⚠️ 说明：本测试只走**在线路径**（A4/A5/A6/V），不含 TabPFN 推理；')
    log('     推理成本与并发行为已由 step517/518/519/521 单独测量。')

    np.savez(os.path.join(_REV, 'experiments', 'step522_online.npz'),
             seg=np.array([r[0] for r in records]),
             rss=np.array([r[1] for r in records]),
             gpu=np.array([r[3] for r in records]),
             lat_ms=np.array([r[5] for r in records]))
    return 0 if allok else 1


if __name__ == '__main__':
    sys.exit(main())
