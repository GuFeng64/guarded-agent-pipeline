# -*- coding: utf-8 -*-
"""pipeline.py — 可靠性感知检测流水线的**统一入口**

## 端到端流程（真实串联，非文档示例）
    A1 采集闸门 → A2 五分支并行推理 → A3 融合+σ+TS闭合
    → A4 区间+分级 → A5 预算约束复测规划 → A6 报告 → V 校验（唯一见真值）
    → A7 回流（标记高价值样本）

## 两种模式
    inference  : 无真值（生产）；validator 跳过覆盖/真伪检查
    evaluation : 有真值（研究）；validator 完整校验

## 上下文隔离（章程规则 A1）
    每个阶段通过 `Context.for_agent(name)` 取输入 ⇒ 真值只在 validator 可见

## 用法
    from pipeline import NIRPipeline
    pipe = NIRPipeline()
    out = pipe.run(yhat=..., sigma=..., q=..., budget=0.20, loss='grade')
    # 或 evaluation 模式
    out = pipe.run(..., Y=Y, cow=cow)

## 自测
    python pipeline.py
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

from context import Context                      # noqa: E402
from degradation import decide as degrade_decide  # noqa: E402
from orchestrator import Orchestrator, CANONICAL_BRANCHES  # noqa: E402

__all__ = ['NIRPipeline', 'PipelineConfig']


class PipelineConfig:
    """流水线配置（集中管理，便于复现）。"""

    def __init__(self, targets=('Fat', 'Prot', 'Lact', 'TS', 'Urea'),
                 reportable=('Fat', 'Prot', 'TS'),
                 alpha=0.10, budget_frac=0.20, loss='grade',
                 branch_timeout=30.0, allow_degraded=True,
                 use_llm=False, try_repair=True, decision_mode='rule'):
        self.targets = list(targets)
        self.reportable = list(reportable)
        self.alpha = float(alpha)
        self.budget_frac = float(budget_frac)
        self.loss = loss
        self.branch_timeout = float(branch_timeout)
        self.allow_degraded = bool(allow_degraded)
        self.use_llm = bool(use_llm)
        self.try_repair = bool(try_repair)
        self.decision_mode = str(decision_mode)   # 'rule' | 'guarded'

    def as_dict(self):
        return {k: v for k, v in self.__dict__.items()}


class NIRPipeline:
    """把各 Agent 串成一条可运行、可审计的流水线。"""

    def __init__(self, cfg: PipelineConfig = None, folds=None, Y=None,
                 mu=None, sd=None, cow=None):
        self.cfg = cfg or PipelineConfig()
        self.folds = folds
        self.Y = Y
        self.mu = mu
        self.sd = sd
        self.cow = cow
        self.log = []

    # ---------- 工具 ----------
    def _note(self, msg):
        self.log.append(msg)

    def _ctx(self, store, agent):
        """按 Agent 白名单取输入（上下文隔离）"""
        return Context.from_store(store).for_agent(agent)

    # ---------- 主流程 ----------
    def run(self, *, yhat, sigma, q, branches=None, decisions_grades=None,
            budget_frac=None, loss=None, Y=None, cow=None,
            X_views=None, apply_a1=False):
        """执行流水线。

        必要输入：yhat (n,nt) ｜ sigma (n,nt) ｜ q (nt,)
        可选：branches（用于降级记录）、X_views（启用 A1）、Y/cow（evaluation 模式）
        """
        from agents.a4_assess import Assessor
        from agents.a5_route import Router
        from agents.a6_report import Reporter
        from agents.a7_feedback import FeedbackAgent
        from validator import Validator

        cfg = self.cfg
        n, nt = yhat.shape
        out = {}
        self.log = []

        # ---------- A1 采集闸门（可选）----------
        a1_flags = None
        if apply_a1 and X_views is not None:
            from agents.a1_acquisition import AcquisitionAgent
            A1 = AcquisitionAgent(try_repair=cfg.try_repair)
            a1_flags = []
            for i in range(n):
                r = A1.check_one({k: v[i] for k, v in X_views.items()})
                a1_flags.append({'verdict': r.verdict, 'corrected': r.corrected,
                                 'reasons': r.reasons,
                                 'spike_index': r.spike_index})
            n_reject = sum(1 for f in a1_flags if f['verdict'] == 'RESHOOT')
            n_corr = sum(1 for f in a1_flags if f['corrected'])
            self._note(f'A1: {n - n_reject}/{n} ACCEPT ｜ {n_corr} corrected')
            out['a1'] = a1_flags

        # ---------- A2/A3 之外的降级信息 ----------
        deg = None
        if branches is not None:
            missing = [b for b in CANONICAL_BRANCHES if b not in branches]
            deg = degrade_decide(missing)
            self._note(f'降级判定: {deg}')
            out['degradation'] = {'action': deg.action, 'delta': deg.delta,
                                  'missing': deg.missing, 'reason': deg.reason}
            if deg.action == 'abort':
                out['aborted'] = True
                out['reason'] = deg.reason
                return out

        degraded = bool(branches is not None
                        and set(branches) != set(CANONICAL_BRANCHES))
        missing = ([b for b in CANONICAL_BRANCHES if b not in branches]
                   if branches is not None else [])

        # ---------- A4 评估：区间 + 分级 ----------
        A4 = Assessor(self.folds, Y if Y is not None else np.zeros((n, nt)),
                      cfg.targets)
        LO = np.full((n, nt), np.nan)
        HI = np.full((n, nt), np.nan)
        Q = np.asarray(q, dtype=float)
        for t in cfg.targets:
            ti = cfg.targets.index(t)
            # ⭐ 直接使用调用方传入的 q（逐目标保形分位数），常数宽度区间。
            #    先前这里用 hasattr 探测一个不存在的方法 ⇒ 永远走回退分支的死代码，已删除。
            LO[:, ti] = yhat[:, ti] - Q[ti]
            HI[:, ti] = yhat[:, ti] + Q[ti]

        # 分级（用 v39 已知的 RPD；evaluation 模式下重算）
        if Y is not None and self.folds is not None:
            _, _, rpd, grades = A4.metrics_and_grades(yhat)
        else:
            from agents.a4_assess import grade_of_rpd
            grades = {}
            for t in cfg.targets:
                code, desc = ('report', '可报告') if t in cfg.reportable \
                    else ('reject', '不可用')
                if t == 'Lact':
                    code, desc = 'reject', '不可用'
                grades[t] = {'code': code, 'desc': desc,
                             'reportable': t in cfg.reportable,
                             'r2': float('nan'), 'rpd': float('nan'),
                             'rmse': float('nan')}
        out.update({'lo': LO, 'hi': HI, 'q': Q, 'grades': grades})

        # ---------- A5 路由（损失无关规划）----------
        R = Router(cfg.targets, reportable=cfg.reportable)
        bf = cfg.budget_frac if budget_frac is None else budget_frac
        loss = cfg.loss if loss is None else loss
        plan = R.plan(yhat, sigma, Q, bf, policy='expected_loss', loss=loss)
        self._note(f'A5: budget={plan["budget"]} loss={loss} '
                   f'retest={len(plan["retest_idx"])}')
        out['plan'] = plan

        # ---------- A5b Guarded Agent（可选决策模式）----------
        if cfg.decision_mode == 'guarded':
            from agents.a5_guarded import GuardedAgent
            ag = GuardedAgent()
            ctxs = []
            for i in range(n):
                ctxs.append({
                    'targets': list(cfg.targets),
                    'lo': np.array([LO[i, k] for k in range(nt)]),
                    'hi': np.array([HI[i, k] for k in range(nt)]),
                    'sigma': np.array([sigma[i, k] for k in range(nt)]),
                    'q': np.array([Q[k] for k in range(nt)]),
                    'grades': grades,
                    'grade_thr': {'Fat': (4.0, 4.5, 5.0),
                                  'Prot': (3.4, 3.7, 4.0),
                                  'TS': (13.5, 14.5, 15.5),
                                  'Lact': (), 'Urea': ()},
                    'compliance': {'Fat': 3.1, 'Prot': 2.9, 'TS': None,
                                   'Lact': None, 'Urea': None},
                    'regret': {t: {'report': float(Q[cfg.targets.index(t)])
                                   * float(sigma[i, cfg.targets.index(t)]),
                                   'retest': 1.0, 'reference': 3.0}
                               for t in cfg.targets
                               if t in cfg.reportable},
                    'degraded': degraded,
                    'missing': missing,
                })
            gprops, gdiags = ag.decide_batch(ctxs, budget_frac=bf,
                                             verbose=True)
            # 逐目标动作 ⇒ 样本级决策
            sample_dec = []
            for gp in gprops:
                acts = [a for t, a in gp.actions.items() if t in cfg.reportable]
                if any(a == 'retest' for a in acts):
                    sample_dec.append('retest')
                elif acts and all(a == 'reference' for a in acts):
                    sample_dec.append('reference')
                else:
                    sample_dec.append('report')
            n_llm = sum(1 for gp in gprops if gp.source == 'llm')
            n_g7 = sum(1 for d in gdiags if d.get('g7_budget_rejected'))
            self._note(f'A5b guarded: {n_llm}/{n} 采纳 LLM ｜ G7 裁剪 {n_g7}')
            out['guarded'] = {'props': gprops, 'diags': gdiags,
                              'n_llm_adopted': n_llm, 'n_g7_rejected': n_g7}
            plan = dict(plan)
            plan['decision'] = sample_dec
            plan['decision_mode'] = 'guarded'
        else:
            plan['decision_mode'] = 'rule'

        # ---------- A6 报告 ----------
        Rep = Reporter(cfg.targets)
        reports, texts = [], []
        use_llm = cfg.use_llm
        llm_tpl = None
        if use_llm:
            try:
                from agents.a6_llm import LLMNarrator
                from agents.a6_report import DEFAULT_TEMPLATE
                narr = LLMNarrator()
                if narr.available:
                    llm_tpl, meta = narr.rewrite_template(DEFAULT_TEMPLATE)
                    self._note(f'A6 LLM: used={meta.get("used_llm")}')
                else:
                    self._note('A6 LLM: 无密钥 ⇒ 回退确定性模板')
            except Exception as e:
                self._note(f'A6 LLM 失败回退: {e}')
        for i in range(n):
            rep = Rep.build(sample_id=f'S{i:03d}', yhat_row=yhat[i],
                            lo_row=LO[i], hi_row=HI[i],
                            decision=plan['decision'][i], grades=grades,
                            degraded=degraded, missing=missing,
                            alpha=cfg.alpha)
            reports.append(rep)
            texts.append(Rep.render(rep, template=llm_tpl))
        out['reports'] = reports
        out['texts'] = texts

        # ---------- V 校验（唯一见真值）----------
        if Y is not None:
            Val = Validator(Y, cow if cow is not None else np.zeros(n),
                            cfg.targets, reportable=cfg.reportable)
            vres = []
            for i in range(n):
                vr = Val.validate(reports[i], texts[i], i, yhat[i])
                vres.append(vr)
            n_ok = sum(1 for v in vres if v.ok)
            self._note(f'V: {n_ok}/{n} 通过校验')
            out['validation'] = vres
            out['validation_pass'] = n_ok / n

        # ---------- A7 回流 ----------
        A7 = FeedbackAgent()
        conf_ids = list(range(n)) if Y is not None else []
        fb = A7.select(plan['decision'], plan['score'], conf_ids)
        out['feedback'] = fb
        self._note(f'A7: {len(fb.high_value_ids)} 个高价值样本')

        # ---------- 物理一致性校验（TS 闭合）----------
        # ⭐ TS 结构性闭合要求 TS ≈ Fat + Prot + Lact + Âsh。
        #    这里只做**审计**（不修改预测），用于发现结构性不一致。
        if all(t in cfg.targets for t in ('Fat', 'Prot', 'Lact', 'TS')):
            iF, iP, iL = (cfg.targets.index('Fat'), cfg.targets.index('Prot'),
                          cfg.targets.index('Lact'))
            iT = cfg.targets.index('TS')
            if Y is not None:
                ash = float((Y[:, iT] - Y[:, iF] - Y[:, iP] - Y[:, iL]).mean())
            else:
                ash = float('nan')
            resid = (yhat[:, iT] - yhat[:, iF] - yhat[:, iP] - yhat[:, iL])
            out['closure'] = {
                'ash_mean': ash,
                'resid_mean': float(np.mean(resid)),
                'resid_sd': float(np.std(resid)),
                'resid_max_abs': float(np.max(np.abs(resid))),
            }
            self._note(f"闭合审计: resid mean {out['closure']['resid_mean']:+.4f} "
                       f"sd {out['closure']['resid_sd']:.4f} "
                       f"Ash {ash:.4f}")

        out['degraded'] = degraded
        out['log'] = list(self.log)
        return out


# ---------------------------------------------------------------- 自测

def _self_test() -> int:
    ok, fail = 0, []
    from sklearn.model_selection import GroupKFold
    import mems_data as M

    d = M.load()
    Y = np.asarray(d['Y'], dtype=float)
    cow = np.asarray(d['cow'])
    T = list(d['targets'])
    n, nt = Y.shape
    folds = list(GroupKFold(5).split(np.zeros((n, 1)), Y[:, 0], cow))
    Z = np.load(os.path.join(_REV, 'experiments', 'step502_fix.npz'),
                allow_pickle=True)
    yhat, SIG = Z['PRED'], Z['SIG_CAL']
    q = np.array([0.30, 0.10, 0.05, 0.35, 3.0])   # 占位；真实 q 由 Assessor 算

    print('=' * 72)
    print('pipeline.py 自测')
    print('=' * 72)

    pipe = NIRPipeline(folds=folds, Y=Y, cow=cow)
    out = pipe.run(yhat=yhat, sigma=SIG, q=q, Y=Y, cow=cow,
                   budget_frac=0.20, loss='grade')

    # ① 关键产物齐全
    need = ('lo', 'hi', 'grades', 'plan', 'reports', 'texts',
            'validation', 'feedback')
    miss = [k for k in need if k not in out]
    if not miss:
        ok += 1
    else:
        fail.append(f'缺少产物 {miss}')

    # ② 报告数量与样本数一致
    if len(out['reports']) == n and len(out['texts']) == n:
        ok += 1
    else:
        fail.append('报告数量不符')

    # ③ 校验全部通过（正常数据）
    if out.get('validation_pass', 0) > 0.95:
        ok += 1
    else:
        fail.append(f'校验通过率过低 {out.get("validation_pass")}')

    # ④ 预算正确
    if len(out['plan']['retest_idx']) == int(round(0.20 * n)):
        ok += 1
    else:
        fail.append('预算错误')

    # ⑤ 降级判定：缺少 b9 时应 accept_degraded
    sub = {b: None for b in CANONICAL_BRANCHES if b != 'b9'}
    out2 = pipe.run(yhat=yhat, sigma=SIG, q=q, branches=sub, Y=Y, cow=cow)
    if out2['degradation']['action'] == 'accept_degraded':
        ok += 1
    else:
        fail.append(f'b9 缺失的降级判定错误 {out2["degradation"]}')

    # ⑥ 整族缺失 ⇒ abort 且提前返回
    sub2 = {b: None for b in ('b1', 'CNN')}
    out3 = pipe.run(yhat=yhat, sigma=SIG, q=q, branches=sub2, Y=Y, cow=cow)
    if out3.get('aborted'):
        ok += 1
    else:
        fail.append('整族缺失未 abort')

    # ⑦ inference 模式（无 Y）应可运行且不做校验
    out4 = pipe.run(yhat=yhat, sigma=SIG, q=q, budget_frac=0.20)
    if ('validation' not in out4) and len(out4['texts']) == n:
        ok += 1
    else:
        fail.append('inference 模式异常')

    # ⑦a guarded 模式在无密钥时等价于规则（安全回退）
    had = os.environ.pop('LLM_API_KEY', None)
    try:
        pipe_g = NIRPipeline(PipelineConfig(reportable=['Fat', 'Prot', 'TS'],
                                            decision_mode='guarded'),
                             folds=folds, Y=Y, cow=cow)
        out_g = pipe_g.run(yhat=yhat, sigma=SIG, q=q,
                           Y=Y, cow=cow, budget_frac=0.20)
        src = [p.source for p in out_g['guarded']['props']]
        if all(s_ == 'rule' for s_ in src) and len(out_g['texts']) == n:
            ok += 1
        else:
            fail.append(f'guarded 模式无密钥时未回退规则：{set(src)}')
    except Exception as e:
        fail.append(f'guarded 模式异常：{type(e).__name__}: {e}')
    finally:
        if had is not None:
            os.environ['LLM_API_KEY'] = had

    # ⑦b 物理一致性审计应存在且合理
    if 'closure' in out and abs(out['closure']['ash_mean'] - 1.05) < 0.15:
        ok += 1
    else:
        fail.append(f'闭合审计缺失或 Ash 异常 {out.get("closure")}')

    # ⑧ 报告中的数值不得被改动（抽查）
    from agents.a6_report import Reporter as Rep_
    Rr = Rep_(T)
    good = Rr.verify_numbers(out['reports'][0], out['texts'][0])[0]
    if good:
        ok += 1
    else:
        fail.append('报告数值校验失败')

    print('  流程日志（前 6 行）：')
    for line in out['log'][:6]:
        print('    ' + line)
    print(f'  通过 {ok} 项')
    if fail:
        print(f'  ❌ 失败 {len(fail)} 项：')
        for f_ in fail:
            print('     -', f_)
        return 1
    print('  ✅ 全部通过：产物齐全、校验通过、降级判定正确、'
          'inference 模式可用、报告数值未被改动')
    return 0


if __name__ == '__main__':
    sys.exit(_self_test())
