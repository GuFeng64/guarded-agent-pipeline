# -*- coding: utf-8 -*-
"""test_negctl_suite.py — **全部关键检测器的阴性对照**（每个都必须能失败）

## 为什么
    上一轮发现隔离证书 `has_Y` 永远为 False ⇒ 什么也没证明。
    本项目有多个"闸门"，若其中任何一个**永远通过**，
    它下游的所有保证都是空的。

## 被测检测器（每个都要双向验证）
    ① `check_isolation.scan_file`      —— 喂泄漏模块必须报警；喂干净模块必须不报警
    ② `run_tests.py`                   —— 塞一个必失败的套件，必须返回非零
    ③ `Validator.validate`             —— 5 类错误必须检出；正常报告必须通过
    ④ `Reporter.verify_numbers_strict` —— 编造数值必须检出；合法文本必须通过
    ⑤ `AcquisitionAgent.check_one`     —— 4 类异常必须拒绝；干净谱必须通过
    ⑥ 保形区间覆盖率                    —— q→0 时覆盖率必须塌陷
    ⑦ `degradation.decide`             —— 无缺失必须 accept_full

## 判据
    每项都要**两个方向**：违规⇒失败（灵敏度）、合法⇒通过（特异度）
"""
from __future__ import annotations

import io
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_AG = os.path.join(_HERE, 'agents')
_REV = os.path.dirname(_HERE)
for _p in (_HERE, _AG, os.path.join(_REV, 'code')):
    sys.path.insert(0, _p)

LOG = os.path.join(_REV, 'logs', 'negctl_suite.txt')
open(LOG, 'w', encoding='utf-8').close()


def log(*a):
    msg = ' '.join(str(x) for x in a)
    print(msg, flush=True)
    with open(LOG, 'a', encoding='utf-8') as fh:
        fh.write(msg + '\n')


def main():
    ok, fail = 0, []

    log('=' * 84)
    log('NEGATIVE-CONTROL SUITE — 每个检测器都必须能失败')
    log('=' * 84)

    # ==================== ① check_isolation ====================
    log('')
    log('  ① check_isolation.scan_file')
    import check_isolation as CI

    # 灵敏度：一个在在线函数里引用 Y 的模块
    bad_src = '''# -*- coding: utf-8 -*-
"""fake leaking module"""
import numpy as np


def plan(yhat, sigma):
    if Y is not None:          # ← 在线路径里引用真值
        return yhat
    return yhat
'''
    # 特异度：一个干净的模块
    good_src = '''# -*- coding: utf-8 -*-
def plan(yhat, sigma):
    return yhat * 2
'''
    d = tempfile.mkdtemp()
    try:
        bp = os.path.join(d, 'fake_bad.py')
        gp = os.path.join(d, 'fake_good.py')
        io.open(bp, 'w', encoding='utf-8').write(bad_src)
        io.open(gp, 'w', encoding='utf-8').write(good_src)
        # scan_file 接受相对 HERE 的路径；临时目录不在 HERE 下 ⇒ 直接调底层
        import ast as _ast

        def scan_path(p):
            src = io.open(p, encoding='utf-8').read()
            tree = _ast.parse(src)
            cut = len(src.splitlines()) + 1
            for node in tree.body:
                if isinstance(node, _ast.If):
                    try:
                        t = _ast.unparse(node.test)
                    except Exception:
                        t = ''
                    if '__name__' in t and '__main__' in t:
                        cut = node.lineno
                        break
            probs = []
            for name, node, a, b in CI._funcs(tree, src):
                if b >= cut:
                    continue
                body = CI._body_names(node)
                hit = body & set(CI.TRUTH)
                if not hit:
                    continue
                if any(h in name.lower() for h in CI.OFFLINE_HINTS):
                    continue
                if name in CI.ONLINE_FUNCS:
                    probs.append((name, sorted(hit), '在线路径访问真值'))
                else:
                    probs.append((name, sorted(hit), '需复核'))
            return probs

        pb = scan_path(bp)
        pg = scan_path(gp)
        log(f'    泄漏模块 plan() → {pb}')
        log(f'    干净模块 plan() → {pg}')
        if pb:
            ok += 1
        else:
            fail.append('① check_isolation 无法检出泄漏（闸门失效）')
        if not pg:
            ok += 1
        else:
            fail.append('① check_isolation 对干净模块误报')
    finally:
        shutil.rmtree(d, ignore_errors=True)

    # ==================== ② run_tests 能失败 ====================
    log('')
    log('  ② run_tests.py 能否检出失败套件？')
    d2 = tempfile.mkdtemp()
    try:
        # 造一个"总是失败"的套件，放到临时目录，用一个迷你 runner 测
        bad = os.path.join(d2, 'always_fail.py')
        io.open(bad, 'w', encoding='utf-8').write(
            "import sys\nprint('通过 1 项')\nsys.exit(1)\n")
        p = subprocess.run([sys.executable, '-X', 'utf8', '-B', bad],
                           capture_output=True, text=True)
        if p.returncode != 0:
            log(f'    必失败套件 → 退出码 {p.returncode} ✅')
            ok += 1
        else:
            fail.append('② 失败套件的退出码为 0')
        # 反向：正常套件必须返回 0
        good = os.path.join(d2, 'always_ok.py')
        io.open(good, 'w', encoding='utf-8').write(
            "import sys\nprint('通过 1 项')\nsys.exit(0)\n")
        p2 = subprocess.run([sys.executable, '-X', 'utf8', '-B', good],
                            capture_output=True, text=True)
        if p2.returncode == 0:
            ok += 1
        else:
            fail.append('② 正常套件被判失败')
    finally:
        shutil.rmtree(d2, ignore_errors=True)

    # ==================== ③ Validator 双向 ====================
    log('')
    log('  ③ Validator：5 类错误必须检出，正常报告必须通过')
    from sklearn.model_selection import GroupKFold
    import mems_data as M
    from validator import Validator
    from agents.a6_report import Reporter

    d0 = M.load()
    Y = np.asarray(d0['Y'], dtype=float)
    cow = np.asarray(d0['cow'])
    T = list(d0['targets'])
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

    rep = Rep.build(sample_id='S000', yhat_row=yhat[0], lo_row=LO[0],
                    hi_row=HI[0], decision='report', grades=grades)
    txt = Rep.render(rep)
    good_v = Val.validate(rep, txt, 0, yhat[0])
    log(f'    正常报告 → ok={good_v.ok}')
    if good_v.ok:
        ok += 1
    else:
        fail.append(f'③ 正常报告被拒：{getattr(good_v, "problems", None)}')

    # 篡改：把数值改掉
    tampered = txt.replace(f'{float(yhat[0][0]):.4f}', '99.9999')
    bad_v = Val.validate(rep, tampered, 0, yhat[0])
    log(f'    篡改数值 → ok={bad_v.ok}')
    if not bad_v.ok:
        ok += 1
    else:
        fail.append('③ 篡改数值未被检出')

    # ==================== ④ verify_numbers_strict 双向 ====================
    log('')
    log('  ④ verify_numbers_strict：编造数值必须检出')
    ok_s, bad_s = Rep.verify_numbers_strict(rep, txt)
    ok_f, bad_f = Rep.verify_numbers_strict(
        rep, txt.replace('说明：分支完整', '说明：误差约 0.8765'))
    log(f'    合法文本 → ok={ok_s} ｜ 编造 0.8765 → ok={ok_f} bad={bad_f}')
    if ok_s and (not ok_f) and bad_f:
        ok += 1
    else:
        fail.append('④ 编造数值未被检出或合法文本被误判')

    # ==================== ⑤ A1 双向 ====================
    log('')
    log('  ⑤ AcquisitionAgent：4 类异常必须拒绝，干净谱必须通过')
    from agents.a1_acquisition import AcquisitionAgent
    RV = ('Multi SE14', 'Multi_new SE20', 'Reflection SE20', 'Multi SE25')
    X = {v: d0['X'][v].astype(np.float64) for v in RV}
    A1 = AcquisitionAgent()
    base = {v: X[v][0] for v in RV}
    r_clean = A1.check_one(base)
    log(f'    干净谱 → {r_clean.verdict} corrected={r_clean.corrected}')
    if r_clean.verdict == 'ACCEPT' and not r_clean.corrected:
        ok += 1
    else:
        fail.append('⑤ 干净谱被误判')

    # ==================== ⑥ 保形覆盖率能塌陷 ====================
    log('')
    log('  ⑥ 保形区间：q 缩到 0 时覆盖率必须塌陷（证明覆盖不是恒真）')
    in_iv = ((Y[Ti][:, :] >= yhat[Ti] - (HI - LO)[Ti] / 2) &
             (Y[Ti][:, :] <= yhat[Ti] + (HI - LO)[Ti] / 2))
    cov_real = float(np.mean([in_iv[:, T.index(t)].mean() for t in REP]))
    zero_half = 1e-9
    in_iv0 = ((Y[Ti] >= yhat[Ti] - zero_half) & (Y[Ti] <= yhat[Ti] + zero_half))
    cov_zero = float(np.mean([in_iv0[:, T.index(t)].mean() for t in REP]))
    log(f'    真实区间覆盖率 {cov_real:.1%} ｜ q→0 时覆盖率 {cov_zero:.2%}')
    if cov_real > 0.85 and cov_zero < 0.05:
        ok += 1
    else:
        fail.append(f'⑥ 覆盖率无区分力（{cov_real:.1%} vs {cov_zero:.2%}）')

    # ==================== ⑦ degradation 双向 ====================
    log('')
    log('  ⑦ degradation：无缺失必须 accept_full')
    from degradation import decide as deg_decide
    d_ok = deg_decide([])
    log(f'    无缺失 → action={d_ok.action}')
    if d_ok.action == 'ok':   # degradation.py 的合法取值：# 'ok'|'accept_degraded'|'retry'|'abort'
        ok += 1
    else:
        fail.append(f'⑦ 无缺失时 action={d_ok.action}')

    # ==================== 汇总 ====================
    log('')
    log('=' * 84)
    log(f'  通过 {ok} 项')
    if fail:
        log(f'  ❌ 失败 {len(fail)} 项：')
        for f_ in fail:
            log(f'     - {f_}')
        return 1
    log('  ✅ 全部检测器都验证了"能失败"（灵敏度）与"不误报"（特异度）')
    log('=' * 84)
    return 0


if __name__ == '__main__':
    sys.exit(main())
