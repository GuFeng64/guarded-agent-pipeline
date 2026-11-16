# -*- coding: utf-8 -*-
"""step523_rss_ab.py — RSS 增长的 A/B 归因（tracemalloc 开销 vs 真实泄漏）

## 背景
    step522 装好 psutil 后测到：Python 堆 +0.15 MB，但 **RSS +430 MB**
    ⇒ 必须查清这 430 MB 是 tracemalloc 的开销，还是系统自身的内存增长

## 实验
    三种配置各跑 2970 条同一流：
      A  baseline            —— 不启动 tracemalloc
      B  tracemalloc on      —— 与 step522 相同
      C  只做报告渲染（不校验）—— 排除 Validator 的影响

## 判据
    · 若 A 的 RSS 增长 << B ⇒ 430 MB 主要是 tracemalloc 开销
    · 若 A 的 RSS 同样增长 ⇒ 系统存在真实内存增长，需定位
    · 若 C 明显小于 A ⇒ Validator 是增长来源
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

LOG = os.path.join(_REV, 'logs', 'step523_rss.txt')
open(LOG, 'w', encoding='utf-8').close()


def log(*a):
    msg = ' '.join(str(x) for x in a)
    print(msg, flush=True)
    with open(LOG, 'a', encoding='utf-8') as fh:
        fh.write(msg + '\n')


def rss():
    import psutil
    return psutil.Process().memory_info().rss / 1e6


def run_stream(mode, n_rounds=10):
    """mode: 'plain' | 'trace' | 'no_validate'"""
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

    if mode == 'trace':
        import tracemalloc
        tracemalloc.start()

    R = Router(T, reportable=REPORTABLE)
    Rep = Reporter(T)
    grades = {}
    Ti = np.concatenate([te for _, te in folds])
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
    Val = None if mode == 'no_validate' else Validator(Y, cow, T,
                                                       reportable=REPORTABLE)
    plan0 = R.plan(yhat, sigma, Q, 0.20, policy='expected_loss', loss='grade')
    DEC = list(plan0['decision'])

    rng = np.random.default_rng(0)
    stream = []
    for rd in range(n_rounds):
        stream.extend([(int(i), rd) for i in rng.permutation(n)])

    # 预热
    for k in range(30):
        i = int(rng.integers(0, n))
        rep = Rep.build(sample_id=f'W{k:03d}', yhat_row=yhat[i], lo_row=LO[i],
                        hi_row=HI[i], decision='dec', grades=grades)
        _ = Rep.render(rep)
    gc.collect()

    m0 = rss()
    samples = []
    for pos, (i, rd) in enumerate(stream):
        rep = Rep.build(sample_id=f'S{i:03d}', yhat_row=yhat[i], lo_row=LO[i],
                        hi_row=HI[i], decision=DEC[i], grades=grades)
        txt = Rep.render(rep)
        if Val is not None:
            _ = Val.validate(rep, txt, i, yhat[i])
        if (pos + 1) % 500 == 0:
            gc.collect()
            samples.append((pos + 1, rss() - m0))
    m1 = rss()
    if mode == 'trace':
        import tracemalloc
        cur, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        return m1 - m0, cur / 1e6, peak / 1e6, samples
    return m1 - m0, None, None, samples


def main():
    log('=' * 88)
    log('RSS A/B ATTRIBUTION (2970 conditions per mode)')
    log('=' * 88)
    log(f'  {"mode":16}{"ΔRSS(MB)":>12}{"tracemalloc cur":>18}{"peak":>10}')
    RES = {}
    for mode in ('no_validate', 'plain', 'trace'):
        gc.collect()
        d_rss, cur, peak, smp = run_stream(mode)
        RES[mode] = (d_rss, cur, peak, smp)
        cs = f'{cur:.1f}' if cur is not None else '-'
        ps = f'{peak:.1f}' if peak is not None else '-'
        log(f'  {mode:16}{d_rss:>+12.1f}{cs:>18}{ps:>10}')
        log(f'     分点：{[round(s[1],1) for s in smp]}')
        gc.collect()

    log('')
    log('=' * 88)
    log('  判读')
    log('=' * 88)
    dv = RES['no_validate'][0]
    dp = RES['plain'][0]
    dt = RES['trace'][0]
    log(f'  不校验：{dv:+.1f} MB ｜ 校验：{dp:+.1f} MB ｜ 校验+tracemalloc：{dt:+.1f} MB')
    if abs(dp - dv) > 50:
        log(f'  ⇒ Validator 是主要来源（差 {dp - dv:+.1f} MB）')
    if dt - dp > 100:
        log(f'  ⇒ **tracemalloc 自身占 {dt - dp:.1f} MB**（追踪开销，非系统泄漏）')
    if abs(dp) < 50:
        log('  ⇒ 系统自身 RSS 增长 < 50 MB，**无实质泄漏**')
    else:
        log(f'  ⇒ ⚠️ 系统自身 RSS 增长 {dp:+.1f} MB，需进一步定位')

    np.savez(os.path.join(_REV, 'experiments', 'step523_rss.npz'),
             modes=np.array(list(RES)),
             drss=np.array([RES[m][0] for m in RES]))
    return 0


if __name__ == '__main__':
    sys.exit(main())
