# -*- coding: utf-8 -*-
"""test_audit_negctl.py — 对抗性自审：**证据是否有阴性对照**

## 为什么要这个审计
    本项目已犯过 5 次"看起来通过、实际没验证"的错误。
    本轮专查三类问题：

    A **阴性对照缺失**：一个"证书/断言"如果**永远不会失败**，它就没证明任何事
       ⇒ 本项目最可疑的是隔离证书 `has_Y=False`：
         子进程本来就不需要 Y，所以它当然是 False。
         必须验证：**如果子进程真的泄漏了 Y，这个证书能否报出来？**

    B **同源比较（tautology）**：拿代码和自己比
       ⇒ a3 自测里"新旧接口一致"：`Aggregator.run` 内部就是调 fit+apply，
         比较二者等于比较自己。
         必须改为与**独立产物**（step502 缓存）比较。

    C **结果被丢弃**：算出来但不检查
       ⇒ step522 在线稳定性里 `_ = Val.validate(...)`，必须断言其通过。

## 本脚本做什么
    为 A / B / C 各建一个**能失败的测试**（阴性对照），并跑一遍。
"""
from __future__ import annotations

import io
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_AG = os.path.join(_HERE, 'agents')
_REV = os.path.dirname(_HERE)
for _p in (_HERE, _AG, os.path.join(_REV, 'code')):
    sys.path.insert(0, _p)

LOG = os.path.join(_REV, 'logs', 'audit_negctl.txt')
open(LOG, 'w', encoding='utf-8').close()


def log(*a):
    msg = ' '.join(str(x) for x in a)
    print(msg, flush=True)
    with open(LOG, 'a', encoding='utf-8') as fh:
        fh.write(msg + '\n')


# ---------------------------------------------------------------- 故意泄漏的 worker
LEAK_MODULE_SRC = '''# -*- coding: utf-8 -*-
"""_leak_probe.py — **故意泄漏**的 worker 模块（阴性对照用）

在模块级加载真值 ⇒ 若隔离证书有效，必须能报出 has_Y=True。
本模块只用于测试，不参与系统。
"""
import os
import numpy as np

# ⭐ 故意在模块级加载真值（模拟"隔离被破坏"）
_Y_PATH = os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), 'experiments', 'step502_fix.npz')
try:
    _z = np.load(_Y_PATH, allow_pickle=True)
    Y = _z['Y']            # ← 泄漏！
    cow = _z['cow']
except Exception:
    Y, cow = None, None


def report_globals():
    return {'has_Y': 'Y' in globals(),
            'Y_not_none': Y is not None,
            'Y_shape': (None if Y is None else tuple(np.asarray(Y).shape))}
'''


def main():
    ok, fail = 0, []

    log('=' * 82)
    log('ADVERSARIAL AUDIT — 证据的阴性对照')
    log('=' * 82)

    # ============ A. 隔离证书的阴性对照 ============
    log('')
    log('  A 隔离证书是否有阴性对照？')
    log('    → 建一个**故意在模块级加载 Y** 的 worker，看证书能否报出来')

    leak_path = os.path.join(_HERE, '_leak_probe.py')
    io.open(leak_path, 'w', encoding='utf-8', newline='').write(LEAK_MODULE_SRC)
    time.sleep(0.1)

    try:
        from process_isolation import IsolatedAgent
        # 用一个白名单里存在的名字（借用 'grade'）来构造，只测证书
        probe = IsolatedAgent('grade', '_leak_probe', 'report_globals',
                             timeout=60)
        cert = probe.certificate or {}
        log(f'    证书：data_modules={cert.get("data_modules")}  '
            f'module_arrays={cert.get("module_arrays")}  pid={cert.get("pid")}')
        # 直接调用拿到更详细的信息
        info = probe.call()
        log(f'    泄漏探针返回：{info}')
        probe.close()
        # 阴性对照成立的条件：探针自己报告 Y is not None
        if info.get('Y_not_none'):
            log('    ✅ 阴性对照有效：探针确实加载到了 Y')
            # 证书必须命中断言：模块级出现了 ndarray
            detected = (cert.get('module_arrays', 0) >= 1)
            if detected:
                log(f'    ✅ 证书正确报出 module_arrays={cert["module_arrays"]}'
                    f'（能检出泄漏）')
                ok += 1
            else:
                log(f'    ❌ 证书未报出泄漏（module_arrays='
                    f'{cert.get("module_arrays")}）'
                    f' ⇒ **证书无区分力，不能作为隔离证据**')
                fail.append('A: 隔离证书无法检出真实泄漏')
        else:
            log('    ⚠️ 探针未能加载到 Y（数据路径可能不对）⇒ 本对照无效')
            fail.append('A: 阴性对照无效（探针未加载到数据）')
    except Exception as e:
        log(f'    ❌ 探针构造失败：{type(e).__name__}: {e}')
        fail.append(f'A: 探针失败 {e}')
    finally:
        try:
            os.remove(leak_path)
        except Exception:
            pass

    # ============ B. a3 的"新旧一致"是否同源 ============
    log('')
    log('  B a3 自测的"新旧接口一致"是否为同源比较？')
    src = io.open(os.path.join(_AG, 'a3_aggregate.py'), encoding='utf-8').read()
    # 看 Aggregator.run 是否调用 fit_aggregator + apply_aggregator_idx
    same_src = ('fit_aggregator' in src) and ('apply_aggregator_idx' in src)
    run_body_start = src.index('    def run(self, preds, apply_ts_closure=True')
    run_body_end = src.index('# ------', run_body_start)
    run_body = src[run_body_start:run_body_end]
    is_tauto = ('fit_aggregator' in run_body) and ('apply_aggregator_idx' in run_body)
    if is_tauto:
        log('    ⚠️ Aggregator.run 内部就是 fit+apply ⇒ 与"新路径"比较是**同源**')
        # ⇒ 改用独立产物对比：step502 的 PRED
        from sklearn.model_selection import GroupKFold
        import mems_data as M
        from agents.a2_branch import BranchAgent
        from agents.a3_aggregate import fit_aggregator, apply_aggregator_idx
        d = M.load()
        Y = np.asarray(d['Y'], dtype=float)
        cow = np.asarray(d['cow'])
        T = list(d['targets'])
        n, nt = Y.shape
        folds = list(GroupKFold(5).split(np.zeros((n, 1)), Y[:, 0], cow))
        B = np.load(os.path.join(_REV, 'experiments',
                                 'step426_4v_vi4_cache.npz'), allow_pickle=True)
        MU = [B[f'f{fi}_mu'] for fi in range(5)]
        SD = [B[f'f{fi}_sd'] for fi in range(5)]
        preds = {}
        for nm in ('b1', 'CNN', 'b7', 'b8', 'b9'):
            ag = BranchAgent(nm, folds)
            preds[nm] = {'tr': [ag.fold_train(fi) for fi in range(5)],
                         'te': [ag.fold_test(fi) for fi in range(5)]}
        fit = fit_aggregator(folds, Y, MU, SD, preds, T,
                             apply_ts_closure=False, cow=cow)
        out = apply_aggregator_idx(fit, folds, preds, MU, SD, n)
        Z = np.load(os.path.join(_REV, 'experiments', 'step502_fix.npz'),
                    allow_pickle=True)
        md = float(np.abs(out['base'] - Z['PRED']).max())
        ms = float(np.abs(out['sigma'] - Z['SIG_CAL']).max())
        log(f'    独立产物对比（step502）：base 最大差 {md:.2e} ｜ '
            f'sigma 最大差 {ms:.2e}')
        if md < 1e-10 and ms < 1e-10:
            log('    ✅ 与独立产物逐元素一致 ⇒ 这条证据是**非同源**的，有效')
            ok += 1
        else:
            fail.append(f'B: 与独立产物不一致（{md:.2e}/{ms:.2e}）')
    else:
        log('    ✅ 非同源比较')
        ok += 1

    # ============ C. 在线稳定性里校验结果是否被丢弃 ============
    log('')
    log('  C 在线稳定性测试是否丢弃了校验结果？')
    p522 = os.path.join(_HERE, 'step522_online_stability.py')
    s2 = io.open(p522, encoding='utf-8').read()
    discarded = '_ = Val.validate(' in s2
    if discarded:
        log('    ⚠️ 确实写成 `_ = Val.validate(...)` ⇒ 校验即使全失败也不会发现')
        # 实测：校验到底通过率如何
        from sklearn.model_selection import GroupKFold
        import mems_data as M
        from validator import Validator
        from agents.a6_report import Reporter
        d = M.load()
        Y = np.asarray(d['Y'], dtype=float)
        cow = np.asarray(d['cow'])
        T = list(d['targets'])
        n, nt = Y.shape
        folds = list(GroupKFold(5).split(np.zeros((n, 1)), Y[:, 0], cow))
        Z = np.load(os.path.join(_REV, 'experiments', 'agent_evaluate.npz'),
                    allow_pickle=True)
        yhat, LO, HI = Z['yhat'], Z['lo'], Z['hi']
        REP = ['Fat', 'Prot', 'TS']
        Ti = np.concatenate([te for _, te in folds])
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
        Rep = Reporter(T)
        Val = Validator(Y, cow, T, reportable=REP)
        n_ok = 0
        for i in range(n):
            rep = Rep.build(sample_id=f'S{i:03d}', yhat_row=yhat[i],
                            lo_row=LO[i], hi_row=HI[i],
                            decision='report', grades=grades)
            txt = Rep.render(rep)
            vr = Val.validate(rep, txt, i, yhat[i])
            n_ok += int(vr.ok)
        log(f'    实测校验通过率：{n_ok}/{n} = {n_ok / n:.1%}')
        if n_ok == n:
            log('    ✅ 实测全部通过 ⇒ 之前的"丢弃"未掩盖问题，但**写法仍是缺陷**')
            ok += 1
        else:
            fail.append(f'C: 校验通过率 {n_ok}/{n}，被 `_ =` 掩盖了')
    else:
        log('    ✅ 未丢弃')
        ok += 1

    # ============ D. 检查阈值有无区分度 ============
    log('')
    log('  D 部分断言阈值是否缺乏区分度？')
    suspects = {
        'process_isolation: ipc_ms < 50': 50.0,
        'a1: 整批通过率 > 0.95': 0.95,
        'pipeline: validation_pass > 0.95': 0.95,
    }
    for k, v in suspects.items():
        log(f'    {k}  → 实测值远低于/高于阈值时不具区分力，'
            f'应在报告中同时给出**实测值与余量**')
    log('    ⚠️ 这是"表述质量"问题，不构成错误，但建议报告余量')
    ok += 1

    # ============ 汇总 ============
    log('')
    log('=' * 82)
    log(f'  通过 {ok} 项')
    if fail:
        log(f'  ❌ 发现 {len(fail)} 项问题：')
        for f_ in fail:
            log(f'     - {f_}')
        return 1
    log('  ✅ 三项证据问题均已定位；A 的阴性对照已建立')
    log('=' * 82)
    return 0


if __name__ == '__main__':
    import time
    sys.exit(main())
