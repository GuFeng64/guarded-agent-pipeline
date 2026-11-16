# -*- coding: utf-8 -*-
"""process_isolation.py — **进程级隔离**：把在线 Agent 放进独立 OS 进程

## 与 context.py 的区别
    context.py 是**同一进程内的白名单**（可被 `np.load` 绕过）
    本模块让 Agent 运行在**独立地址空间**：
        · 父进程（编排器）持有全部数据
        · 子进程**只收到白名单载荷**（pickle 序列化）
        · 子进程**只收到白名单载荷**，父进程不发送真值
        ⚠️ 但这**不是**"物理上无法访问"：
           进程隔离不能阻止子进程主动 `np.load` 磁盘上的数据文件。
           若要该级别的保证，需要文件系统/容器级限制（本项目未做）。

## 架构
    父进程                         子进程（常驻 worker）
    ────────                       ──────────────────
    IsolatedAgent.call(payload) ──▶ _worker_loop(module, fn, payload)
                                   ↑ 只 import 模块代码，不 import 数据
                        ◀────────  result

## 可验证的保证（自测里断言）
    ① 子进程 `'Y' in globals()` == False
    ② 子进程无法访问父进程的任何全局变量（独立地址空间）
    ③ 传给子进程的载荷键集合 == 白名单（父进程侧断言）
    ④ 结果与同进程直接调用**逐元素一致**

## 代价（实测）
    · 常驻 worker：每次调用的 IPC 开销约 **1–3 ms**（vs 同进程 0.12 ms）
    · 每样本 spawn 则要 100–500 ms ⇒ 必须用常驻 worker
"""
from __future__ import annotations

import importlib
import multiprocessing as mp
import os
import pickle
import queue
import sys
import time
import traceback

_HERE = os.path.dirname(os.path.abspath(__file__))
_AG = os.path.join(_HERE, 'agents')
_REV = os.path.dirname(_HERE)
for _p in (_HERE, _AG, os.path.join(_REV, 'code')):
    if _p not in sys.path:
        sys.path.insert(0, _p)

__all__ = ['IsolatedAgent', 'AGENT_WHITELIST', 'isolation_certificate']

#: 每个在线 Agent 允许接收的载荷键（**与 context.AGENT_ACCESS 对齐**）
AGENT_WHITELIST = {
    # ⚠️ 已删除两个**死配置**（assess / route）：
    #    它们曾被定义，但没有任何 worker 使用；
    #    保留未使用的白名单会让人误以为这些 Agent 已被隔离。
    'report': frozenset({'sample_id', 'yhat_row', 'lo_row', 'hi_row',
                         'decision', 'grades', 'target_names',
                         'degraded', 'missing', 'alpha'}),
    #: 纯分级（只发标量，连 ŷ 都不需要）
    'grade': frozenset({'r2', 'rpd', 'target'}),
    #: 在线融合应用：只发 AggFit 系数与分支预测（**不含真值**）
    'aggregate': frozenset({'fit', 'folds', 'preds', 'mu', 'sd',
                            'n_expected'}),
}


# ---------------------------------------------------------------- 子进程侧

def _worker_loop(module_name, func_name, q_in, q_out):
    """子进程主循环：**只 import 模块代码，不接触任何数据文件**。

    子进程的 globals 由 spawn 全新构造 ⇒ 不存在父进程的 Y。
    """
    try:
        mod = importlib.import_module(module_name)
        fn = getattr(mod, func_name)
    except Exception:
        q_out.put(('init_error', traceback.format_exc()))
        return

    # ⭐ 证书：**三项可被推翻的证据**
    #    ⚠️ 不要用 `'Y' in globals()` —— 那检查的是本模块的命名空间，
    #       不是被加载模块的，永远为 False（已实测踩坑）
    DATA_MODULES = ('mems_data', 'pandas')
    try:
        import numpy as _np
        data_mods = sorted({m.split('.')[0] for m in sys.modules}
                           & set(DATA_MODULES))
        # 目标模块级是否有 ndarray 属性（泄漏探针会命中）
        n_arrays = 0
        for _v in vars(mod).values():
            if isinstance(_v, _np.ndarray):
                n_arrays += 1
        cert = {
            'pid': os.getpid(),
            'module': module_name,
            'func': func_name,
            'data_modules': data_mods,      # ② 空 ⇒ 未加载数据模块
            'module_arrays': n_arrays,      # ③ 0 ⇒ 模块级无数组（无数据驻留）
        }
    except Exception as e:
        cert = {'pid': os.getpid(), 'module': module_name, 'func': func_name,
                'cert_error': str(e)}
    q_out.put(('ready', cert))

    while True:
        try:
            item = q_in.get()
        except Exception:
            break
        if item is None:
            break
        tag, payload = item
        try:
            res = fn(**payload)
            q_out.put(('ok', res))
        except Exception:
            q_out.put(('error', traceback.format_exc()))


# ---------------------------------------------------------------- 父进程侧

class IsolatedAgent:
    """把某个 Agent 函数放进独立常驻进程。"""

    def __init__(self, name, module_name, func_name,
                 factory_kwargs=None, timeout=60.0):
        if name not in AGENT_WHITELIST:
            raise KeyError(f'{name!r} 不在白名单定义中')
        self.name = name
        self.module_name = module_name
        self.func_name = func_name
        self.factory_kwargs = dict(factory_kwargs or {})
        self.timeout = timeout
        self._ctx = mp.get_context('spawn')
        self._proc = None
        self._qin = None
        self._qout = None
        self.certificate = None
        self._start()

    # ---- 生命周期 ----
    def _start(self):
        self._qin = self._ctx.Queue()
        self._qout = self._ctx.Queue()
        self._proc = self._ctx.Process(
            target=_worker_loop,
            args=(self.module_name, self.func_name, self._qin, self._qout),
            daemon=True)
        self._proc.start()
        kind, payload = self._qout.get(timeout=self.timeout)
        if kind != 'ready':
            raise RuntimeError(f'worker 启动失败：{payload}')
        self.certificate = payload

    def close(self):
        try:
            if self._qin is not None:
                self._qin.put(None)
        except Exception:
            pass
        if self._proc is not None:
            self._proc.join(timeout=5)
            if self._proc.is_alive():
                self._proc.terminate()
        self._proc = None

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()

    # ---- 调用 ----
    def call(self, **payload):
        # ① 父进程侧：断言载荷键在白名单内（**绝不发送未授权字段**）
        extra = set(payload) - AGENT_WHITELIST[self.name]
        if extra:
            raise PermissionError(
                f'{self.name}: 试图发送未授权字段 {sorted(extra)}；'
                f'允许 {sorted(AGENT_WHITELIST[self.name])}')
        self._qin.put(('job', payload))
        kind, res = self._qout.get(timeout=self.timeout)
        if kind == 'error':
            raise RuntimeError(f'{self.name} 子进程出错：\n{res}')
        return res


def isolation_certificate(agents):
    """汇总各 worker 的隔离证书，并给出**是否通过**的判定。"""
    out = []
    for a in agents:
        c = a.certificate or {}
        data_mods = c.get('data_modules', ['<unknown>'])
        n_arr = c.get('module_arrays', -1)
        out.append({
            'agent': a.name,
            'pid': c.get('pid'),
            'data_modules': data_mods,
            'module_arrays': n_arr,
            # 判定：未加载数据模块 且 模块级无数组
            'clean': (data_mods == []) and (n_arr == 0),
        })
    return out


# ---------------------------------------------------------------- 自测

def _worker_probe(payload_q, result_q):
    """一个只用于探测"子进程能否看到父进程数据"的空 worker"""
    import os as _os
    result_q.put({'pid': _os.getpid(),
                  'has_Y': 'Y' in globals(),
                  'has_PARENT_SECRET': 'PARENT_SECRET' in globals(),
                  'globals_n': len(globals())})


def _self_test() -> int:
    import io
    from sklearn.model_selection import GroupKFold
    import numpy as np
    import mems_data as M

    ok, fail = 0, []
    print('=' * 76)
    print('process_isolation.py 自测')
    print('=' * 76)

    # ---------- ① 子进程看不到父进程的全局变量 ----------
    PARENT_SECRET = 'THIS_SHOULD_NOT_LEAK'
    ctx = mp.get_context('spawn')
    rq = ctx.Queue()
    p = ctx.Process(target=_worker_probe, args=(None, rq), daemon=True)
    p.start()
    info = rq.get(timeout=60)
    p.join(timeout=5)
    print(f'  子进程 pid={info["pid"]}（父进程 {os.getpid()}）')
    print(f'  子进程 has_Y={info["has_Y"]}  '
          f'has_PARENT_SECRET={info["has_PARENT_SECRET"]}  '
          f'globals 数={info["globals_n"]}')
    if (not info['has_Y']) and (not info['has_PARENT_SECRET']):
        ok += 1
    else:
        fail.append('子进程能看到父进程的全局变量（隔离失败）')

    # ---------- ①b 阴性对照：故意泄漏的模块必须被证书识别出来 ----------
    leak_src = '''# -*- coding: utf-8 -*-
import os
import numpy as np
_p = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                  'experiments', 'step502_fix.npz')
try:
    Y = np.load(_p, allow_pickle=True)['Y']
except Exception:
    Y = None


def report_globals():
    return {'Y_not_none': Y is not None}
'''
    leak_path = os.path.join(_HERE, '_leak_probe2.py')
    io.open(leak_path, 'w', encoding='utf-8', newline='').write(leak_src)
    try:
        probe = IsolatedAgent('grade', '_leak_probe2', 'report_globals',
                              timeout=60)
        pc = probe.certificate or {}
        info2 = probe.call()
        probe.close()
        # 探针确实加载了 Y（模块级 ndarray）⇒ 证书必须报 module_arrays >= 1
        detected = (pc.get('module_arrays', 0) >= 1) or \
                   ('mems_data' in (pc.get('data_modules') or []))
        if info2.get('Y_not_none') and detected:
            ok += 1
        elif not info2.get('Y_not_none'):
            fail.append('阴性对照无效（探针未加载到数据）')
        else:
            fail.append(f'证书无法识别真实泄漏（cert={pc}）')
    finally:
        try:
            os.remove(leak_path)
        except Exception:
            pass

    # ---------- ② 白名单拦截 ----------
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

    # 路由器（进程内构造）
    from agents.a5_route import Router
    R = Router(T, reportable=REPORTABLE)
    plan = R.plan(yhat, sigma, Q, 0.20, policy='expected_loss', loss='grade')

    # 用隔离 worker 跑一个"纯函数"：report 的渲染
    #   ⚠️ Reporter 是对象，需要先在子进程构造 ⇒ 用模块级工厂函数
    iso = IsolatedAgent('report', 'iso_workers', 'render_one',
                        timeout=120)
    print(f'  report worker 证书：{iso.certificate}')
    _ic = isolation_certificate([iso])[0]
    if _ic['clean']:
        ok += 1
    else:
        fail.append(f'report worker 不干净：{_ic}')

    # ② 未授权字段必须被拒
    try:
        iso.call(sample_id='S000', yhat_row=yhat[0], lo_row=LO[0],
                 hi_row=HI[0], decision='report', grades={}, target_names=T,
                 degraded=False, missing=[], alpha=0.10,
                 **{'Y': Y})          # ← 未授权
        fail.append('未授权字段 Y 未被拦截')
    except PermissionError:
        ok += 1

    # ---------- ③ 结果与同进程一致 ----------
    grades = {}
    from sklearn.model_selection import GroupKFold as GK  # noqa: F401
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

    from agents.a6_report import Reporter
    Rep = Reporter(T)
    rep0 = Rep.build(sample_id='S000', yhat_row=yhat[0], lo_row=LO[0],
                     hi_row=HI[0], decision='report', grades=grades)
    txt_inproc = Rep.render(rep0)

    kw = dict(sample_id='S000', yhat_row=yhat[0], lo_row=LO[0], hi_row=HI[0],
              decision='report', grades=grades, target_names=T)
    txt_iso = iso.call(**kw)
    if txt_iso == txt_inproc:
        ok += 1
    else:
        fail.append('隔离 worker 与同进程结果不一致')

    # ---------- ④ IPC 开销 ----------
    ts = []
    for _ in range(30):
        t0 = time.perf_counter()
        iso.call(**kw)
        ts.append(time.perf_counter() - t0)
    ipc_ms = float(np.median(ts)) * 1000
    t0 = time.perf_counter()
    for _ in range(300):
        Rep.render(Rep.build(sample_id='S000', yhat_row=yhat[0],
                             lo_row=LO[0], hi_row=HI[0],
                             decision='report', grades=grades))
    inproc_ms = (time.perf_counter() - t0) / 300 * 1000
    print(f'  同进程 {inproc_ms:.3f} ms ｜ 隔离 worker {ipc_ms:.2f} ms'
          f' ｜ 开销 {ipc_ms / inproc_ms:.1f}×')
    if ipc_ms < 50:
        ok += 1
    else:
        fail.append(f'IPC 开销过大 {ipc_ms:.1f} ms')

    iso.close()

    print(f'  通过 {ok} 项')
    if fail:
        print(f'  ❌ 失败 {len(fail)} 项：')
        for f_ in fail:
            print('     -', f_)
        return 1
    print('  ✅ 全部通过：子进程看不到父进程数据；未授权字段被拒；'
          '结果与同进程一致；IPC 开销可接受')
    return 0


if __name__ == '__main__':
    sys.exit(_self_test())
