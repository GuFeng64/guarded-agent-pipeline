# -*- coding: utf-8 -*-
"""orchestrator.py — DAG 执行 + 分支级并行调度 + 超时 + 结构化失败

## 设计要点（对应 AGENT_TASKS.md T0.2 判据）
    ① 5 个分支全部返回时，结果必须与串行**逐元素一致**
    ② 注入超时后，返回**结构化失败原因**，不静默降级
    ③ 归并顺序按 **分支 ID**（`CANONICAL_BRANCHES`），不按完成时间
       ⇒ 并行不得引入非确定性（AGENT_CHARTER 规则 C2）

## 关键实现选择
    · 用 ThreadPoolExecutor：分支工作是 numpy/缓存读取，GIL 影响小
    · **先全部 submit，再按规范顺序逐个 result(timeout=...)**
      ⇒ 既保持并行，又保证归并顺序确定
    · 每个分支返回 BranchResult，含 ok/error/elapsed，失败不抛异常到上层
    · 超时后调用 cancel()，但已在运行的线程无法强杀 —— 文档中说明

## 用法
    orch = Orchestrator(timeout=30.0)
    res = orch.run_branches(['b1','CNN','b7','b8','b9'], fn)   # fn(name) -> ndarray
    stack, degraded, missing = orch.assemble(res)

## 自测
    python orchestrator.py
"""
from __future__ import annotations

import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutTimeout
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

__all__ = ['BranchResult', 'StageResult', 'Orchestrator',
           'CANONICAL_BRANCHES', 'CANONICAL_FAMILIES']

#: 规范分支顺序（归并必须按此顺序，**不得**按完成时间）
CANONICAL_BRANCHES: Tuple[str, ...] = ('b1', 'CNN', 'b7', 'b8', 'b9')

#: 分支到族（用于"整族丢失 ⇒ 不可降级"的判定）
CANONICAL_FAMILIES: Dict[str, str] = {
    'b1': 'b1', 'CNN': 'CNN', 'b7': 'TabPFN', 'b8': 'TabPFN', 'b9': 'TabPFN'}


# ---------------------------------------------------------------- 结果结构

@dataclass
class BranchResult:
    """单个分支的执行结果（结构化，失败不抛异常）。"""
    name: str
    ok: bool
    pred: Optional[Any] = None
    error: Optional[str] = None          # 'timeout' / 'exception' / 'invalid'
    detail: Optional[str] = None
    elapsed: float = 0.0

    def __repr__(self):
        if self.ok:
            return f'BranchResult({self.name}, ok, {self.elapsed:.2f}s)'
        return f'BranchResult({self.name}, FAILED[{self.error}], {self.elapsed:.2f}s)'


@dataclass
class StageResult:
    """一个 DAG 阶段的结果。"""
    name: str
    ok: bool
    value: Optional[Any] = None
    error: Optional[str] = None
    elapsed: float = 0.0


# ---------------------------------------------------------------- 编排器

class Orchestrator:
    def __init__(self, timeout: float = 30.0, max_workers: Optional[int] = None):
        self.timeout = float(timeout)
        self.max_workers = max_workers or len(CANONICAL_BRANCHES)
        self.log: List[str] = []

    # -- 内部 --
    def _note(self, msg: str):
        self.log.append(msg)

    # -- 单阶段（DAG 节点）--
    def run_stage(self, name: str, fn: Callable[[], Any],
                  timeout: Optional[float] = None) -> StageResult:
        t0 = time.perf_counter()
        try:
            val = fn()
            el = time.perf_counter() - t0
            self._note(f'stage {name}: ok ({el:.2f}s)')
            return StageResult(name=name, ok=True, value=val, elapsed=el)
        except Exception as e:
            el = time.perf_counter() - t0
            self._note(f'stage {name}: FAILED {type(e).__name__}: {e}')
            return StageResult(name=name, ok=False, error=type(e).__name__,
                               value=f'{e}\n{traceback.format_exc()}', elapsed=el)

    # -- 分支并行 --
    def run_branches(self, names: Sequence[str],
                     fn: Callable[[str], Any],
                     timeout: Optional[float] = None) -> Dict[str, BranchResult]:
        """并行执行分支；**按规范顺序归并**，失败返回结构化结果。

        fn(name) -> ndarray ；抛异常/超时都会被抓成 BranchResult(ok=False)
        """
        to = self.timeout if timeout is None else float(timeout)
        self._note(f'run_branches: {list(names)} timeout={to}s '
                   f'workers={self.max_workers}')

        order = [n for n in CANONICAL_BRANCHES if n in names]
        unknown = [n for n in names if n not in CANONICAL_BRANCHES]
        if unknown:
            raise ValueError(f'未知分支名 {unknown}；规范顺序 {CANONICAL_BRANCHES}')

        out: Dict[str, BranchResult] = {}
        t_start = time.perf_counter()
        with ThreadPoolExecutor(max_workers=self.max_workers) as ex:
            futs = {n: ex.submit(self._call, fn, n) for n in order}
            # ⭐ 按**规范顺序**读取结果 ⇒ 归并确定（规则 C2）
            for n in order:
                t0 = time.perf_counter()
                try:
                    pred, el = futs[n].result(timeout=to)
                    out[n] = BranchResult(name=n, ok=True, pred=pred, elapsed=el)
                    self._note(f'  {n}: ok ({el:.2f}s)')
                except FutTimeout:
                    futs[n].cancel()
                    el = time.perf_counter() - t0
                    out[n] = BranchResult(name=n, ok=False, error='timeout',
                                          detail=f'>{to}s', elapsed=el)
                    self._note(f'  {n}: TIMEOUT (>{to}s)')
                except Exception as e:
                    el = time.perf_counter() - t0
                    out[n] = BranchResult(name=n, ok=False, error='exception',
                                          detail=f'{type(e).__name__}: {e}',
                                          elapsed=el)
                    self._note(f'  {n}: EXCEPTION {type(e).__name__}: {e}')
        self._note(f'run_branches done in {time.perf_counter() - t_start:.2f}s')
        return out

    @staticmethod
    def _call(fn, name):
        t0 = time.perf_counter()
        return fn(name), time.perf_counter() - t0

    # -- 归并 --
    def assemble(self, res: Dict[str, BranchResult],
                 allow_degraded: bool = True) -> Tuple[List[Any], bool, List[str]]:
        """按规范顺序归并成功分支。

        returns (preds_in_order, degraded, missing_names)
        若整族（TabPFN 三变体全丢）缺失 ⇒ 抛 RuntimeError（不可降级）
        """
        order = [n for n in CANONICAL_BRANCHES if n in res]
        ok = [n for n in order if res[n].ok]
        missing = [n for n in order if not res[n].ok]

        if not missing:
            return [res[n].pred for n in ok], False, []

        fams_lost = {CANONICAL_FAMILIES[n] for n in missing}
        if 'TabPFN' in fams_lost and len([n for n in ok
                                          if CANONICAL_FAMILIES[n] == 'TabPFN']) == 0:
            raise RuntimeError(
                f'整个 TabPFN 族缺失（缺 {missing}）⇒ 融合退化 +0.0258，不可降级')

        if not allow_degraded:
            raise RuntimeError(f'分支缺失 {missing} 且不允许降级')

        self._note(f'degraded: missing={missing}')
        return [res[n].pred for n in ok], True, missing


# ---------------------------------------------------------------- 自测

def _self_test() -> int:
    import numpy as np
    ok, fail = 0, []

    # ---- 判据 ①：并行结果与串行逐元素一致 ----
    def make_fn(sleeps):
        def fn(name):
            time.sleep(sleeps.get(name, 0.0))
            rng = np.random.default_rng(hash(name) % (2 ** 32))
            return rng.normal(size=(7, 3))
        return fn

    sleeps = {'b1': 0.20, 'CNN': 0.05, 'b7': 0.15, 'b8': 0.02, 'b9': 0.10}
    fn = make_fn(sleeps)

    orch = Orchestrator(timeout=10.0)
    par = orch.run_branches(list(CANONICAL_BRANCHES), fn)

    serial = {n: fn(n) for n in CANONICAL_BRANCHES}

    same = all(par[n].ok and np.array_equal(par[n].pred, serial[n])
               for n in CANONICAL_BRANCHES)
    if same:
        ok += 1
    else:
        fail.append('判据①：并行与串行结果不一致')

    # ---- 判据 ③：归并顺序确定（与完成顺序无关）----
    stack, deg, miss = orch.assemble(par)
    ref = [serial[n] for n in CANONICAL_BRANCHES]
    if len(stack) == 5 and all(np.array_equal(a, b) for a, b in zip(stack, ref)):
        ok += 1
    else:
        fail.append('判据③：归并顺序不是规范顺序')

    # ---- 判据 ②：超时被抓成结构化失败 ----
    def slow_fn(name):
        if name == 'b8':
            time.sleep(2.0)
        return np.zeros((7, 3))

    orch2 = Orchestrator(timeout=0.5)
    res2 = orch2.run_branches(list(CANONICAL_BRANCHES), slow_fn)
    if (not res2['b8'].ok and res2['b8'].error == 'timeout'
            and all(res2[n].ok for n in CANONICAL_BRANCHES if n != 'b8')):
        ok += 1
    else:
        fail.append(f'判据②：超时未被正确捕获（b8={res2["b8"]}）')

    # ---- 判据 ④：降级标志与缺失列表正确 ----
    try:
        stack2, deg2, miss2 = orch2.assemble(res2)
        if deg2 and miss2 == ['b8'] and len(stack2) == 4:
            ok += 1
        else:
            fail.append(f'判据④：降级标志/缺失列表错误 deg={deg2} miss={miss2}')
    except Exception as e:
        fail.append(f'判据④：assemble 抛异常 {e}')

    # ---- 判据 ⑤：整族缺失 ⇒ 不可降级 ----
    def fam_fn(name):
        if name in ('b7', 'b8', 'b9'):
            raise ValueError('boom')
        return np.zeros((7, 3))

    orch3 = Orchestrator(timeout=5.0)
    res3 = orch3.run_branches(list(CANONICAL_BRANCHES), fam_fn)
    try:
        orch3.assemble(res3)
        fail.append('判据⑤：整族缺失未报错（应不可降级）')
    except RuntimeError:
        ok += 1

    # ---- 判据 ⑥：异常被抓成结构化失败（不抛到上层）----
    if res3['b7'].error == 'exception' and 'boom' in (res3['b7'].detail or ''):
        ok += 1
    else:
        fail.append(f'判据⑥：异常未被结构化记录 b7={res3["b7"]}')

    # ---- 判据 ⑦：未知分支名应报错 ----
    try:
        orch.run_branches(['b1', 'nope'], fn)
        fail.append('判据⑦：未知分支名未报错')
    except ValueError:
        ok += 1

    print('=' * 72)
    print('orchestrator.py 自测')
    print('=' * 72)
    print(f'  通过 {ok} 项')
    if fail:
        print(f'  ❌ 失败 {len(fail)} 项：')
        for f_ in fail:
            print('     -', f_)
        return 1
    print('  ✅ 全部通过：并行≡串行；超时结构化；归并按规范顺序；整族缺失不可降级')
    return 0


if __name__ == '__main__':
    sys.exit(_self_test())
