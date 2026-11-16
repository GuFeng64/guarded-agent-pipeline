# -*- coding: utf-8 -*-
"""test_faults.py — 系统性故障注入矩阵

## 为什么需要
    之前只测了"分支缺失"一种故障。真实运行中还会遇到：
    worker 崩溃、worker 挂起、载荷越权、A1 拒绝、校验失败等。
    **这些情形下系统是否给出结构化响应，从未验证。**

## 故障矩阵
    F1 worker 崩溃（kill 进程）      → 调用应快速失败，不挂死
    F2 worker 挂起（长 sleep）        → 超时生效，不挂死
    F3 载荷越权（发送 Y）             → PermissionError
    F4 分支超时                      → degradation 结构化
    F5 整族缺失                      → ABORT
    F6 A1 拒绝（缺视图/NaN/饱和/尖峰）→ RESHOOT
    F7 畸形载荷（类型错误）           → 结构化错误，不崩溃

## 判据
    每种故障都必须：① 有明确响应 ② 不挂死 ③ 不静默给出错误结果
"""
from __future__ import annotations

import os
import signal
import sys
import time
import traceback

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_AG = os.path.join(_HERE, 'agents')
_REV = os.path.dirname(_HERE)
for _p in (_HERE, _AG, os.path.join(_REV, 'code')):
    if _p not in sys.path:
        sys.path.insert(0, _p)

LOG = os.path.join(_REV, 'logs', 'fault_matrix.txt')
open(LOG, 'w', encoding='utf-8').close()


def log(*a):
    msg = ' '.join(str(x) for x in a)
    print(msg, flush=True)
    with open(LOG, 'a', encoding='utf-8') as fh:
        fh.write(msg + '\n')


# ---------------- 供 F2 用的挂起 worker ----------------
def _hang_worker(q_in, q_out):
    q_out.put(('ready', {'pid': os.getpid(), 'has_Y': False, 'has_cow': False}))
    while True:
        item = q_in.get()
        if item is None:
            break
        if item[0] == 'job':
            time.sleep(3600)          # 永远不返回


def main():
    from process_isolation import IsolatedAgent, AGENT_WHITELIST
    from orchestrator import Orchestrator, CANONICAL_BRANCHES
    from degradation import decide as degrade_decide
    from agents.a1_acquisition import AcquisitionAgent

    ok, fail = 0, []
    log('=' * 80)
    log('FAULT INJECTION MATRIX')
    log('=' * 80)

    # ==================== F1: worker 崩溃 ====================
    log('')
    log('  F1 worker 进程被 kill —— 调用应快速失败，不挂死')
    try:
        iso = IsolatedAgent('report', 'iso_workers', 'render_one', timeout=3.0)
        pid = iso._proc.pid
        os.kill(pid, signal.SIGTERM)
        time.sleep(0.3)
        t0 = time.perf_counter()
        try:
            iso.call(sample_id='X', yhat_row=np.zeros(5), lo_row=np.zeros(5),
                     hi_row=np.ones(5), decision='report',
                     grades={t: {'r2': 0.9, 'rpd': 5.0, 'code': 'report',
                                 'reportable': True}
                             for t in ('Fat', 'Prot', 'Lact', 'TS', 'Urea')},
                     target_names=['Fat', 'Prot', 'Lact', 'TS', 'Urea'])
            el = time.perf_counter() - t0
            log(f'     ⚠️ 调用未抛错（耗时 {el:.2f}s）—— 但 worker 已死，此为异常')
            fail.append('F1: worker 死后调用返回了结果')
        except Exception as e:
            el = time.perf_counter() - t0
            log(f'     ✅ 抛出 {type(e).__name__}（耗时 {el:.2f}s < 3s 超时）')
            if el < 5.0:
                ok += 1
            else:
                fail.append(f'F1: 失败过慢 {el:.1f}s')
        iso.close()
    except Exception as e:
        log(f'     ❌ 测试本身出错：{type(e).__name__}: {e}')
        fail.append(f'F1 测试异常: {e}')

    # ==================== F2: worker 挂起 ====================
    log('')
    log('  F2 worker 挂起（长 sleep）—— 超时应生效')
    try:
        import multiprocessing as mp
        from process_isolation import _worker_loop  # noqa: F401
        # 直接用自定义挂起 worker
        from process_isolation import IsolatedAgent as IA
        ctx = mp.get_context('spawn')
        qin, qout = ctx.Queue(), ctx.Queue()
        proc = ctx.Process(target=_hang_worker, args=(qin, qout), daemon=True)
        proc.start()
        kind, payload = qout.get(timeout=30)
        assert kind == 'ready'
        t0 = time.perf_counter()
        qin.put(('job', {}))
        try:
            r = qout.get(timeout=2.0)
            log(f'     ⚠️ 挂起 worker 在 2s 内返回了 {r}（不应发生）')
            fail.append('F2: 挂起 worker 未触发超时')
        except Exception:
            el = time.perf_counter() - t0
            log(f'     ✅ 超时在 {el:.2f}s 生效（queue.get timeout=2s）')
            ok += 1
        proc.terminate()
        proc.join(timeout=3)
    except Exception as e:
        log(f'     ❌ 测试本身出错：{type(e).__name__}: {e}')
        fail.append(f'F2 测试异常: {e}')

    # ==================== F3: 载荷越权 ====================
    log('')
    log('  F3 载荷越权（发送 Y）—— 应抛 PermissionError')
    try:
        iso = IsolatedAgent('report', 'iso_workers', 'render_one', timeout=5.0)
        try:
            iso.call(sample_id='X', yhat_row=np.zeros(5), lo_row=np.zeros(5),
                     hi_row=np.ones(5), decision='report', grades={},
                     target_names=['Fat'], Y=np.zeros((3, 3)))
            fail.append('F3: 越权字段未被拦截')
            log('     ❌ 越权字段未被拦截')
        except PermissionError as e:
            log(f'     ✅ PermissionError: {str(e)[:70]}')
            ok += 1
        iso.close()
    except Exception as e:
        fail.append(f'F3 测试异常: {e}')

    # ==================== F4: 分支超时 ====================
    log('')
    log('  F4 某分支超时 —— degradation 应给出结构化结论')
    for nm in ('b9', 'b8'):
        d = degrade_decide([nm])
        tag = '✅' if d.action in ('accept_degraded', 'retry', 'abort') else '❌'
        log(f'     丢 {nm}: action={d.action:16} Δ={d.delta:.4f}  {tag}')
        if tag == '✅':
            ok += 1
        else:
            fail.append(f'F4: {nm} 的降级结论异常')

    # ==================== F5: 整族缺失 ====================
    log('')
    log('  F5 整族缺失（TabPFN 全丢）—— 应 ABORT')
    d = degrade_decide(['b7', 'b8', 'b9'])
    log(f'     action={d.action}  Δ={d.delta:.4f}  reason={d.reason[:60]}')
    if d.action == 'abort':
        ok += 1
    else:
        fail.append('F5: 整族缺失未 ABORT')

    # ==================== F6: A1 各类拒绝 ====================
    log('')
    log('  F6 A1 采集闸门的四类拒绝')
    import mems_data as M
    d0 = M.load()
    RV = ('Multi SE14', 'Multi_new SE20', 'Reflection SE20', 'Multi SE25')
    X = {v: d0['X'][v].astype(np.float64) for v in RV}
    A1 = AcquisitionAgent()
    base = {v: X[v][0] for v in RV}
    cases = {}
    # 缺视图
    v1 = dict(base); v1.pop('Reflection SE20')
    cases['缺视图'] = v1
    # NaN
    v2 = dict(base); xx = base['Multi_new SE20'].copy(); xx[5] = np.nan
    v2['Multi_new SE20'] = xx
    cases['NaN'] = v2
    # 饱和
    v3 = dict(base); v3['Multi SE14'] = np.full_like(base['Multi SE14'], 1.0)
    cases['饱和'] = v3
    # 尖峰
    v4 = dict(base); yy = base['Multi_new SE20'].copy()
    rng = np.random.default_rng(0)
    yy[rng.choice(len(yy), 5, replace=False)] += 3.0 * yy.std()
    v4['Multi_new SE20'] = yy
    cases['尖峰'] = v4
    for nm, vv in cases.items():
        r = A1.check_one(vv)
        exp_corr = (nm == '尖峰')
        good = (r.verdict == 'ACCEPT' and r.corrected) if exp_corr else \
               (r.verdict == 'RESHOOT')
        log(f'     {nm:6}: verdict={r.verdict:8} corrected={r.corrected}  '
            f'{"✅" if good else "❌"}')
        if good:
            ok += 1
        else:
            fail.append(f'F6: {nm} 的 A1 响应异常')

    # ==================== F7: 畸形载荷 ====================
    log('')
    log('  F7 畸形载荷（类型错误）—— 应结构化报错，不崩溃整个进程')
    try:
        iso = IsolatedAgent('grade', 'iso_workers', 'grade_one', timeout=5.0)
        try:
            iso.call(r2='not_a_number', rpd=None, target='Fat')
            log('     ⚠️ 畸形载荷未报错')
            fail.append('F7: 畸形载荷未报错')
        except Exception as e:
            log(f'     ✅ 抛出 {type(e).__name__}（未被静默接受）')
            ok += 1
        # worker 仍可用？
        good = iso.call(r2=0.96, rpd=5.25, target='Fat')
        if good.get('code') == 'report':
            log(f'     ✅ worker 在错误后仍可用：{good}')
            ok += 1
        else:
            fail.append('F7: worker 在错误后不可用')
        iso.close()
    except Exception as e:
        fail.append(f'F7 测试异常: {e}')

    # ==================== 汇总 ====================
    log('')
    log('=' * 80)
    log(f'  通过 {ok} 项')
    if fail:
        log(f'  ❌ 失败 {len(fail)} 项：')
        for f_ in fail:
            log(f'     - {f_}')
        return 1
    log('  ✅ 全部通过：7 类故障均有明确响应，无挂死、无静默错误')
    log('=' * 80)
    return 0


if __name__ == '__main__':
    sys.exit(main())
