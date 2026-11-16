# -*- coding: utf-8 -*-
"""isolated_pipeline.py — 进程级隔离的流水线（与同进程版本结果一致）

## 目的
    证明"进程级隔离"不仅原理可行，而且**能整条流水线跑通、结果不变**。

## 架构
    父进程（持有 Y、yhat、sigma、q）
      │ 只发白名单载荷
      ├─▶ report worker（独立进程）：渲染报告文本
      └─▶ grade worker （独立进程）：按 RPD 分级
    ⭐ 三个 worker 的证书 `module_arrays == 0` 且 `data_modules == []`
       ⚠️ 这是**载荷隔离**，不是文件系统隔离（worker 仍可读磁盘）

## 与 pipeline.py 的关系
    · pipeline.py  —— 同进程版本（快，用于研究/离线评估）
    · 本模块        —— 进程隔离版本（用于"真值不可见"的强保证）
    ⇒ **两者输出必须逐字符一致**（自测断言）

## 用法
    from isolated_pipeline import IsolatedPipeline
    ip = IsolatedPipeline(targets, reportable)
    out = ip.run(yhat, lo, hi, decisions, grades)   # 逐样本
    ip.close()
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

from process_isolation import IsolatedAgent, isolation_certificate  # noqa: E402

__all__ = ['IsolatedPipeline']


class IsolatedPipeline:
    """A6 报告 + A4 分级 全部在独立进程完成。"""

    def __init__(self, targets, reportable=None, timeout=60.0):
        self.targets = list(targets)
        self.reportable = list(reportable or ['Fat', 'Prot', 'TS'])
        self._report = IsolatedAgent('report', 'iso_workers', 'render_one',
                                     timeout=timeout)
        self._grade = IsolatedAgent('grade', 'iso_workers', 'grade_one',
                                    timeout=timeout)
        self._agg = IsolatedAgent('aggregate', 'iso_workers',
                                  'apply_fusion', timeout=timeout)

    # ---- 证书 ----
    def certificates(self):
        return isolation_certificate([self._report, self._grade,
                                      self._agg])

    # ---- 分级（隔离）----
    def grade(self, r2, rpd, target):
        # ⚠️ grade_one 的白名单键是 assess 的；此处只发纯标量
        return self._grade.call(r2=float(r2), rpd=float(rpd), target=str(target))

    # ---- 报告（隔离）----
    def report(self, *, sample_id, yhat_row, lo_row, hi_row, decision,
               grades, degraded=False, missing=None, alpha=0.10):
        return self._report.call(
            sample_id=str(sample_id), yhat_row=np.asarray(yhat_row),
            lo_row=np.asarray(lo_row), hi_row=np.asarray(hi_row),
            decision=str(decision), grades=grades,
            target_names=self.targets, degraded=bool(degraded),
            missing=list(missing or []), alpha=float(alpha))

    # ---- 融合应用（隔离）----
    def aggregate(self, fit, folds, preds, mu, sd, n_expected):
        return self._agg.call(fit=fit, folds=folds, preds=preds,
                              mu=mu, sd=sd, n_expected=int(n_expected))

    def close(self):
        self._report.close()
        self._grade.close()
        self._agg.close()

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()


# ---------------------------------------------------------------- 自测

def _self_test() -> int:
    from sklearn.model_selection import GroupKFold
    import mems_data as M
    from agents.a6_report import Reporter

    ok, fail = 0, []
    print('=' * 76)
    print('isolated_pipeline.py 自测')
    print('=' * 76)

    d = M.load()
    Y = np.asarray(d['Y'], dtype=float)
    cow = np.asarray(d['cow'])
    T = list(d['targets'])
    n, nt = Y.shape
    folds = list(GroupKFold(5).split(np.zeros((n, 1)), Y[:, 0], cow))
    Z = np.load(os.path.join(_REV, 'experiments', 'agent_evaluate.npz'),
                allow_pickle=True)
    yhat, LO, HI = Z['yhat'], Z['lo'], Z['hi']
    REPORTABLE = ['Fat', 'Prot', 'TS']

    # 分级（父进程算好，作为载荷发给 worker）
    Ti = np.concatenate([te for _, te in folds])
    grades, rpd_true = {}, {}
    for t in T:
        ti = T.index(t)
        e = yhat[Ti, ti] - Y[Ti, ti]
        rmse = float(np.sqrt(np.mean(e ** 2)))
        rpd = float(Y[Ti, ti].std() / (rmse + 1e-12))
        r2 = float(1 - np.sum(e ** 2)
                   / np.sum((Y[Ti, ti] - Y[Ti, ti].mean()) ** 2))
        rpd_true[t] = rpd
        code = 'report' if rpd > 3 else ('good' if rpd > 2.5
                                        else ('screening' if rpd > 2 else 'reject'))
        grades[t] = {'r2': r2, 'rpd': rpd, 'code': code,
                     'reportable': code in ('report', 'good')}

    ip = IsolatedPipeline(T, REPORTABLE)
    certs = ip.certificates()
    print('  隔离证书：')
    for c in certs:
        print(f'    {c["agent"]:10} pid={c["pid"]}  '
              f'data_modules={c["data_modules"]}  '
              f'module_arrays={c["module_arrays"]}  clean={c["clean"]}')
    # ① 所有 worker 都看不到真值
    if all(c['clean'] for c in certs):
        ok += 1
    else:
        fail.append('有 worker 证书未通过（加载了数据模块或模块级有数组）')
    # ② PID 与父进程不同
    if all(c['pid'] != os.getpid() for c in certs):
        ok += 1
    else:
        fail.append('worker 与父进程同 PID（未真正隔离）')

    # ③ 隔离报告 == 同进程报告（逐字符）
    Rep = Reporter(T)
    K = 20
    same = 0
    for i in range(K):
        txt_ip = ip.report(sample_id=f'S{i:03d}', yhat_row=yhat[i],
                           lo_row=LO[i], hi_row=HI[i], decision='report',
                           grades=grades)
        rep = Rep.build(sample_id=f'S{i:03d}', yhat_row=yhat[i],
                        lo_row=LO[i], hi_row=HI[i], decision='report',
                        grades=grades)
        txt_p = Rep.render(rep)
        same += int(txt_ip == txt_p)
    print(f'  逐字符一致：{same}/{K}')
    if same == K:
        ok += 1
    else:
        fail.append(f'隔离与同进程结果不一致（{same}/{K}）')

    # ④ 隔离分级 == 同进程分级
    from agents.a4_assess import grade_of_rpd
    gsame = 0
    for t in T:
        g = ip.grade(r2=grades[t]['r2'], rpd=rpd_true[t], target=t)
        exp = grade_of_rpd(rpd_true[t])
        gsame += int(g['code'] == exp[0])
    print(f'  分级一致：{gsame}/{len(T)}')
    if gsame == len(T):
        ok += 1
    else:
        fail.append(f'分级不一致（{gsame}/{len(T)}）')

    # ④b A3 在线应用在隔离 worker 里复现相同结果
    from agents.a3_aggregate import fit_aggregator, apply_aggregator_idx
    from agents.a2_branch import BranchAgent
    from sklearn.model_selection import GroupKFold as _GK
    folds2 = list(_GK(5).split(np.zeros((n, 1)), Y[:, 0], cow))
    preds = {}
    for nm in ('b1', 'CNN', 'b7', 'b8', 'b9'):
        ag = BranchAgent(nm, folds2)
        preds[nm] = {'tr': [ag.fold_train(fi) for fi in range(5)],
                     'te': [ag.fold_test(fi) for fi in range(5)]}
    B2 = np.load(os.path.join(_REV, 'experiments',
                              'step426_4v_vi4_cache.npz'), allow_pickle=True)
    MU2 = [B2[f'f{fi}_mu'] for fi in range(5)]
    SD2 = [B2[f'f{fi}_sd'] for fi in range(5)]
    fit2 = fit_aggregator(folds2, Y, MU2, SD2, preds, T,
                          apply_ts_closure=True, cow=cow)
    out_iso = ip.aggregate(fit2, folds2, preds, MU2, SD2, n)
    out_loc = apply_aggregator_idx(fit2, folds2, preds, MU2, SD2, n)
    same_agg = (np.allclose(out_iso['yhat'], out_loc['yhat'], atol=1e-12)
                and np.allclose(out_iso['sigma'], out_loc['sigma'], atol=1e-12))
    if same_agg:
        ok += 1
    else:
        fail.append('A3 隔离 worker 与本地结果不一致')

    # ⑤ 吞吐：100 条样本的隔离 vs 同进程
    import time
    N = 100
    t0 = time.perf_counter()
    for i in range(N):
        ip.report(sample_id=f'T{i:03d}', yhat_row=yhat[i % n],
                  lo_row=LO[i % n], hi_row=HI[i % n], decision='report',
                  grades=grades)
    t_iso = time.perf_counter() - t0
    t0 = time.perf_counter()
    for i in range(N):
        rep = Rep.build(sample_id=f'T{i:03d}', yhat_row=yhat[i % n],
                        lo_row=LO[i % n], hi_row=HI[i % n],
                        decision='report', grades=grades)
        _ = Rep.render(rep)
    t_in = time.perf_counter() - t0
    print(f'  {N} 条：同进程 {t_in*1000:.2f} ms ｜ 隔离 {t_iso*1000:.2f} ms'
          f' ｜ 开销 {(t_iso-t_in)/N*1000:.3f} ms/样本')
    if (t_iso - t_in) / N < 0.02:      # 每样本 < 20 ms
        ok += 1
    else:
        fail.append('隔离开销过大')

    ip.close()

    print(f'  通过 {ok} 项')
    if fail:
        print(f'  ❌ 失败 {len(fail)} 项：')
        for f_ in fail:
            print('     -', f_)
        return 1
    print('  ✅ 全部通过：worker 独立 PID 且看不到真值；报告/分级与同进程逐字符一致；'
          '开销可接受')
    return 0


if __name__ == '__main__':
    sys.exit(_self_test())
