"""第三輪（外部意見 2026-10-23）：
1. 校正模型精簡（去掉平方項與性別交互作用、加入「自報體重尾數 0／5」），另建殘差變異模型 → 每人 P(實測 BMI ≥ 門檻)；
   校正前後差值用配對 bootstrap 算 CI。
2. 實測策略的取捨前緣（2021–2023 驗證期）：對稱 ±k、不對稱（門檻下方範圍為上方 2 倍，事前定）、機率式（P 落在 [p, 1−p] 才實測）。
3. Kitagawa 分解：子群粗率差 = BMI／條件分布差（組成）＋ 層內誤判率差（率），附 bootstrap CI。
4. 補充：手術版 Table 1、Table 2 比值的 CI 與 2021–2023 單一週期的人數、週期趨勢斜率。

資料、規則與 PSU 重抽權重 WB 沿用 analysis.py 前段。輸出 out/r3_*.csv、out/r3_summary.json。
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import norm
import statsmodels.api as sm

HERE = Path(__file__).resolve().parent
src = (HERE / "analysis.py").read_text()
exec(compile(src[:src.index("t1 = [summary(r)")], "analysis.py", "exec"))
N = len(d); w = d.w.values; Wf = WB.astype(np.float64)
pctl = lambda bs: np.percentile(bs, [2.5, 97.5])
TH = {"AOM": np.where(d.com.values, 27.0, 30.0), "MBS": np.where(d.met.values, 30.0, 35.0)}
COND = {"AOM": d.com.values, "MBS": d.met.values}
tr, te = (d.cycle != "2021-2023").values, (d.cycle == "2021-2023").values
out = {}


def rate(num, den, W=None):
    if W is None:
        return 100 * (w @ (num & den)) / (w @ den)
    return 100 * (W @ (num & den).astype(float)) / (W @ den.astype(float))


# ---------------------------------------------------------------- 1. 模型
heap = np.isin(d.WHD020.values % 10, [0, 5]).astype(float)
X = pd.DataFrame({"const": 1.0, "self_bmi": d.bmi_self, "age10": (d.RIDAGEYR - 50) / 10, "women": (d.RIAGENDR == 2).astype(float),
                  "black": (d.RIDRETH3 == 4).astype(float), "hispanic": d.RIDRETH3.isin([1, 2]).astype(float),
                  "asian": (d.RIDRETH3 == 6).astype(float), "other_race": (d.RIDRETH3 == 7).astype(float), "heaped_weight": heap})
fit = sm.WLS(d.bmi_meas.values[tr], X.values[tr], weights=w[tr]).fit()
mu = X.values @ fit.params
res = d.bmi_meas.values - mu
# 變異模型：log 殘差平方對同一組共變項（訓練期），σ_i = exp(預測/2) × 平滑校正
vfit = sm.WLS(np.log(res[tr] ** 2 + 1e-6), X.values[tr], weights=w[tr]).fit()
lv = X.values @ vfit.params
adj = np.average(res[tr] ** 2, weights=w[tr]) / np.average(np.exp(lv[tr]), weights=w[tr])
sigma = np.sqrt(np.exp(lv) * adj)
coef = pd.DataFrame({"model": "mean", "term": X.columns, "coef": fit.params, "lo": fit.params - 1.96 * fit.bse, "hi": fit.params + 1.96 * fit.bse})
vcoef = pd.DataFrame({"model": "log variance", "term": X.columns, "coef": vfit.params, "lo": vfit.params - 1.96 * vfit.bse, "hi": vfit.params + 1.96 * vfit.bse})
pd.concat([coef, vcoef]).to_csv(OUT / "r3_model_coef.csv", index=False)
z_te = res[te] / sigma[te]
perf = []
for lab, m in [("training 2013–2018", tr), ("validation 2021–2023", te)]:
    e_raw = d.bmi_self.values[m] - d.bmi_meas.values[m]; e_cor = mu[m] - d.bmi_meas.values[m]
    perf.append({"set": lab, "n": int(m.sum()), "mean_error_self": np.average(e_raw, weights=w[m]), "mean_error_corrected": np.average(e_cor, weights=w[m]),
                 "mae_self": np.average(np.abs(e_raw), weights=w[m]), "mae_corrected": np.average(np.abs(e_cor), weights=w[m]),
                 "sd_residual": np.sqrt(np.average((d.bmi_meas.values[m] - mu[m]) ** 2, weights=w[m]))})
pd.DataFrame(perf).to_csv(OUT / "r3_model_perf.csv", index=False)
# 機率模型的校準（驗證期）：預測區間覆蓋率
cover = {f"pi{int(q * 100)}": float(np.average(np.abs(z_te) < norm.ppf(0.5 + q / 2), weights=w[te])) for q in (0.5, 0.8, 0.95)}
out["model"] = {"r2": fit.rsquared, "coverage_validation": cover}

# 校正前後（配對 bootstrap 差值）
cal = []
for r, f in RULES.items():
    if r not in ("AOM", "MBS"):
        continue
    m = d[f"{r}_meas"].values; s_raw = d[f"{r}_self"].values; s_cor = f(mu, d)
    a, b_ = rate(m != s_raw, te), rate(m != s_cor, te)
    A, B = rate(m != s_raw, te, Wf), rate(m != s_cor, te, Wf)
    cal.append({"rule": r, "self_reported": a, "self_lo": pctl(A)[0], "self_hi": pctl(A)[1], "corrected": b_, "cor_lo": pctl(B)[0], "cor_hi": pctl(B)[1],
                "difference": b_ - a, "diff_lo": pctl(B - A)[0], "diff_hi": pctl(B - A)[1],
                "measured_only_self": rate(m & ~s_raw, te), "self_only_self": rate(~m & s_raw, te),
                "measured_only_corrected": rate(m & ~s_cor, te), "self_only_corrected": rate(~m & s_cor, te)})
pd.DataFrame(cal).to_csv(OUT / "r3_calibration.csv", index=False)

# ---------------------------------------------------------------- 2. 取捨前緣（驗證期）
fr = []
for r in ("AOM", "MBS"):
    m = d[f"{r}_meas"].values; s_rule = d[f"{r}_self"].values; T = TH[r]; dlt = d.bmi_self.values - T
    p_elig = 1 - norm.cdf((T - mu) / sigma)
    strategies = [("Symmetric ± k", "k", np.round(np.arange(0, 4.01, 0.25), 2), lambda k: (dlt > -k) & (dlt < k)),
                  ("Asymmetric (−2k to +k)", "k", np.round(np.arange(0, 3.01, 0.125), 3), lambda k: (dlt > -2 * k) & (dlt < k)),
                  ("Probability-based", "p", [0.5, 0.4, 0.3, 0.25, 0.2, 0.15, 0.1, 0.075, 0.05, 0.035, 0.025, 0.015, 0.01, 0.005, 0.002, 0.001],
                   lambda p: (p_elig >= p) & (p_elig <= 1 - p) if p < 0.5 else np.zeros(N, bool))]
    for name, par, grid, zone_fn in strategies:
        for v in grid:
            z = zone_fn(v)
            final = np.where(z, m, s_rule if name != "Probability-based" else (p_elig >= 0.5))
            meas, resd = rate(z, te), rate(m != final, te)
            MB, RB = rate(z, te, Wf), rate(m != final, te, Wf)
            fr.append({"rule": r, "strategy": name, "param": par, "value": v, "measured_pct": meas, "measured_lo": pctl(MB)[0], "measured_hi": pctl(MB)[1],
                       "residual": resd, "res_lo": pctl(RB)[0], "res_hi": pctl(RB)[1],
                       "res_measured_only": rate(m & ~final, te), "res_self_only": rate(~m & final, te)})
fr = pd.DataFrame(fr); fr.to_csv(OUT / "r3_frontier.csv", index=False)
# 操作點：達到殘餘誤判 ≤ 3%、2%、1% 所需的最小實測比例（線性內插於前緣）
ops = []
for (r, st), g in fr.groupby(["rule", "strategy"]):
    g = g.sort_values("measured_pct")
    for target in (3, 2, 1):
        ok = g[g.residual <= target]
        if len(ok):
            ops.append({"rule": r, "strategy": st, "target_residual": target, "measured_pct": ok.measured_pct.iloc[0], "at": f"{ok.param.iloc[0]} = {ok.value.iloc[0]}",
                        "residual": ok.residual.iloc[0]})
pd.DataFrame(ops).to_csv(OUT / "r3_operating_points.csv", index=False)
print("frontier done", flush=True)

# ---------------------------------------------------------------- 3. Kitagawa 分解
def strata(r):
    b = np.clip(np.floor(d.bmi_meas.values), 15, 50).astype(int)
    return b * 2 + COND[r].astype(int)
GROUPS = {
    "Hispanic vs non-Hispanic White": (np.isin(d.RIDRETH3.values, [1, 2]), d.RIDRETH3.values == 3),
    "Non-Hispanic Black vs non-Hispanic White": (d.RIDRETH3.values == 4, d.RIDRETH3.values == 3),
    "Non-Hispanic Asian vs non-Hispanic White": (d.RIDRETH3.values == 6, d.RIDRETH3.values == 3),
    "Women vs men": (d.RIAGENDR.values == 2, d.RIAGENDR.values == 1),
    "≥ 65 vs 18–39 years": (d.RIDAGEYR.values >= 65, d.RIDAGEYR.values < 40),
    "40–64 vs 18–39 years": ((d.RIDAGEYR.values >= 40) & (d.RIDAGEYR.values < 65), d.RIDAGEYR.values < 40)}


def kita(Wm, gmask, rmask, S, dis):
    """Wm: (R, n) 權重；回傳 crude 差、組成、率（百分點），各為長度 R。"""
    def comp(mask):
        den = Wm @ (S * mask[:, None]); num = Wm @ (S * (mask * dis)[:, None])
        p = den / den.sum(1, keepdims=True)
        with np.errstate(invalid="ignore", divide="ignore"):
            rr = num / den
        return p, rr
    pg, rg = comp(gmask.astype(float)); pr, rr_ = comp(rmask.astype(float))
    rg = np.where(np.isnan(rg), rr_, rg); rr_ = np.where(np.isnan(rr_), rg, rr_)
    rg, rr_ = np.nan_to_num(rg), np.nan_to_num(rr_)
    crude = 100 * ((pg * rg).sum(1) - (pr * rr_).sum(1))
    compn = 100 * ((pg - pr) * (rg + rr_) / 2).sum(1)
    return crude, compn, crude - compn
kt = []
for r in ("AOM", "MBS"):
    k = strata(r); ks = np.unique(k); S = (k[:, None] == ks[None, :]).astype(np.float64)
    dis = (d[f"{r}_meas"].values != d[f"{r}_self"].values).astype(float)
    for name, (g, ref) in GROUPS.items():
        c0, p0, r0 = kita(w[None, :], g, ref, S, dis)
        cb, pb, rb = kita(Wf, g, ref, S, dis)
        share = pb / cb * 100
        kt.append({"rule": r, "comparison": name, "crude_diff": c0[0], "crude_lo": pctl(cb)[0], "crude_hi": pctl(cb)[1],
                   "composition": p0[0], "comp_lo": pctl(pb)[0], "comp_hi": pctl(pb)[1],
                   "within_stratum": r0[0], "rate_lo": pctl(rb)[0], "rate_hi": pctl(rb)[1],
                   "composition_share_pct": 100 * p0[0] / c0[0] if abs(c0[0]) > 0.3 else np.nan})
pd.DataFrame(kt).to_csv(OUT / "r3_kitagawa.csv", index=False)
print("kitagawa done", flush=True)

# ---------------------------------------------------------------- 4. 補充：手術版特徵表、Table 2 追加、週期趨勢
disc = d.MBS_meas.values != d.MBS_self.values
cols = {"All": np.ones(N, bool), "Concordant": ~disc, "Discordant": disc}
rows = [{"characteristic": "Participants, n", **{c: f"{int(cm.sum()):,}" for c, cm in cols.items()}},
        {"characteristic": "Weighted population, millions", **{c: f"{w[cm].sum() / 1e6:.1f}" for c, cm in cols.items()}}]
def addp(label, mask):
    rows.append({"characteristic": label, **{c: f"{100 * (w @ (mask & cm)) / (w @ cm):.1f} ({int((mask & cm).sum()):,})" for c, cm in cols.items()}})
def addm(label, v):
    rows.append({"characteristic": label, **{c: f"{np.average(v[cm], weights=w[cm]):.1f} ({np.sqrt(np.average((v[cm] - np.average(v[cm], weights=w[cm])) ** 2, weights=w[cm])):.1f})".replace("-", "−") for c, cm in cols.items()}})
addm("Age, years, mean (SD)", d.RIDAGEYR.values.astype(float))
addp("Women, %", d.RIAGENDR.values == 2)
for lab, codes in [("Non-Hispanic White", [3]), ("Non-Hispanic Black", [4]), ("Hispanic", [1, 2]), ("Non-Hispanic Asian", [6]), ("Other or multiracial", [7])]:
    addp(f"{lab}, %", np.isin(d.RIDRETH3.values, codes))
addm("Measured BMI, kg/m², mean (SD)", d.bmi_meas.values)
for lab, lo, hi in [("< 30", 0, 30), ("30 to < 35", 30, 35), ("35 to < 40", 35, 40), ("≥ 40", 40, 999)]:
    addp(f"Measured BMI {lab} kg/m², %", (d.bmi_meas.values >= lo) & (d.bmi_meas.values < hi))
addm("Self-reported minus measured BMI, kg/m², mean (SD)", (d.bmi_self - d.bmi_meas).values)
addp("Diabetes or hypertension, %", d.met.values)
pd.DataFrame(rows).to_csv(OUT / "r3_table1_mbs.csv", index=False)

t2x = []
for r in ("AOM", "MBS"):
    m, s = d[f"{r}_meas"].values, d[f"{r}_self"].values
    ratio = (w @ (m & ~s)) / (w @ (~m & s)); rb = (Wf @ (m & ~s).astype(float)) / (Wf @ (~m & s).astype(float))
    w4 = w * 4; W4 = Wf * 4   # 2021–2023 單一週期：MEC 權重不除以 4
    pop = lambda mask: (w4 @ (mask & te)) / 1e6
    popb = lambda mask: pctl((W4 @ (mask & te).astype(float)) / 1e6)
    row = {"rule": r, "ratio": ratio, "ratio_lo": pctl(rb)[0], "ratio_hi": pctl(rb)[1]}
    for k, mask in [("pop2123_total", np.ones(N, bool)), ("pop2123_eligible_measured", m), ("pop2123_discordant", m != s), ("pop2123_measured_only", m & ~s), ("pop2123_self_only", ~m & s)]:
        row[k] = pop(mask); row[k + "_lo"], row[k + "_hi"] = popb(mask)
    t2x.append(row)
pd.DataFrame(t2x).to_csv(OUT / "r3_table2_extra.csv", index=False)

MID = {"2013-2014": 2014, "2015-2016": 2016, "2017-2018": 2018, "2021-2023": 2022.6}   # 2021–23：2021 年 8 月至 2023 年 8 月，中點約 2022.6
yr = d.cycle.map(MID).values.astype(float)
tr_rows = []
for r in ("AOM", "MBS"):
    y = (d[f"{r}_meas"].values != d[f"{r}_self"].values).astype(float)
    Xt = np.column_stack([np.ones(N), (yr - 2018) / 10])
    def slope(ww):
        A = Xt.T @ (Xt * ww[:, None]); b = Xt.T @ (y * ww)
        return 100 * np.linalg.solve(A, b)[1]
    sb = np.array([slope(Wf[i]) for i in range(REPS)])
    tr_rows.append({"rule": r, "slope_pp_per_10y": slope(w), "lo": pctl(sb)[0], "hi": pctl(sb)[1]})
pd.DataFrame(tr_rows).to_csv(OUT / "r3_trend.csv", index=False)

# 亞裔專用切點（手術，ASMBS／IFSO 2022：亞裔 BMI ≥ 27.5 應提供手術；PMC 全文的 > 加底線＝≥，2026-10-23 查證）
asian = d.RIDRETH3.values == 6
mbs_as = lambda b: b >= 27.5
v = rate(mbs_as(d.bmi_meas.values) != mbs_as(d.bmi_self.values), asian); vb = rate(mbs_as(d.bmi_meas.values) != mbs_as(d.bmi_self.values), asian, Wf)
out["asian_mbs_specific"] = {"discordant": v, "lo": pctl(vb)[0], "hi": pctl(vb)[1], "n": int(asian.sum())}

# 圖 1 象限：只在「無條件」者（門檻單一）計算
quad = {}
for r, T in (("AOM", 30), ("MBS", 35)):
    nc = ~COND[r]; a_, b_ = d.bmi_meas.values >= T, d.bmi_self.values >= T
    quad[r] = {"threshold": T, "meas_ge_self_lt": rate(a_ & ~b_, nc), "meas_lt_self_ge": rate(~a_ & b_, nc), "n": int(nc.sum())}
out["quadrants_no_condition"] = quad
json.dump(out, open(OUT / "r3_summary.json", "w"), indent=1, default=float)
print(json.dumps(out, indent=1, default=float))
print(pd.DataFrame(cal).round(2).to_string())
print(pd.DataFrame(ops).round(2).to_string())
print(pd.DataFrame(kt).round(2).to_string())
print(pd.DataFrame(t2x).round(2).to_string()); print(pd.DataFrame(tr_rows).round(2))
