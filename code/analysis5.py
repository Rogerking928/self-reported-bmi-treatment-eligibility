"""第五輪（外部意見 2026-10-23）：
1. 公平性的絕對與相對差距：各族群相對於非西裔白人的差（pp）與比（倍），只用自報 vs 不對稱規則（k = 1），附 bootstrap CI。
2. 不含種族的實測策略：以不含種族的模型（自報 BMI、年齡、性別、體重尾數 0／5）估計每人 P(實測 BMI ≥ 門檻)，
   機率落在 [p, 1 − p] 才實測；以「週期互換」交叉擬合（2013–18 擬合套到 2021–23，反之亦然），避免同一資料擬合又評估；
   p 調到與不對稱規則（k = 1）相同的總實測比例，比較各族群殘餘誤判與差距。另一個簡單版：體重尾數為 0／5 者用 k = 1.5、其他人 k 調到總實測比例相同。
3. 針對性灌水：自報 BMI 在門檻下 X kg/m² 內者，有 p 比例把體重報到剛好跨過門檻（BMI = 門檻 + 0.1）；比較 −2 到 +1 與 −2 到 +2 兩種實測範圍。
   另報均勻多報 2 kg 時兩方向互相抵銷的情形（合格盛行率看似正確、個人層級仍誤判）。

資料、規則與 WB 沿用 analysis.py 前段，replicate 權重依 analysis4 在週期內 re-rake。輸出 out/r5_*.csv。
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.api as sm

HERE = Path(__file__).resolve().parent
src = (HERE / "analysis.py").read_text()
exec(compile(src[:src.index("t1 = [summary(r)")], "analysis.py", "exec"))
N = len(d); w = d.w.values; Wf = WB.astype(np.float64)
cyc = d.cycle.values
for c in np.unique(cyc):
    m = cyc == c
    Wf[:, m] *= (w[m].sum() / Wf[:, m].sum(1))[:, None]
pctl = lambda bs: np.percentile(bs, [2.5, 97.5], axis=0)
TH = {"AOM": np.where(d.com.values, 27.0, 30.0), "MBS": np.where(d.met.values, 30.0, 35.0)}
ALLM = np.ones(N, bool)
RACE = [("Non-Hispanic White", [3]), ("Non-Hispanic Black", [4]), ("Hispanic", [1, 2]), ("Non-Hispanic Asian", [6]), ("Other or multiracial", [7])]
G = {lab: np.isin(d.RIDRETH3.values, codes) for lab, codes in RACE}


def rate(num, den, W=None):
    num = num & den
    if W is None:
        return 100 * (w @ num) / (w @ den)
    return 100 * (W @ num.astype(float)) / (W @ den.astype(float))


# ---------------------------------------------------------------- 不含種族的機率（週期互換交叉擬合）
heap = np.isin(d.WHD020.values % 10, [0, 5]).astype(float)
X = np.column_stack([np.ones(N), d.bmi_self, (d.RIDAGEYR - 50) / 10, (d.RIAGENDR == 2), heap]).astype(float)
folds = [(d.cycle != "2021-2023").values, (d.cycle == "2021-2023").values]
mu, sig = np.zeros(N), np.zeros(N)
zfit = {}
for k_, (a, b) in enumerate([(folds[0], folds[1]), (folds[1], folds[0])]):
    f = sm.WLS(d.bmi_meas.values[a], X[a], weights=w[a]).fit(); r_ = d.bmi_meas.values - X @ f.params
    v = sm.WLS(np.log(r_[a] ** 2 + 1e-6), X[a], weights=w[a]).fit(); lv = X @ v.params
    s_ = np.sqrt(np.exp(lv) * np.average(r_[a] ** 2, weights=w[a]) / np.average(np.exp(lv[a]), weights=w[a]))
    z = r_[a] / s_[a]; o = np.argsort(z)
    zfit[k_] = (z[o], np.cumsum(w[a][o]) / w[a].sum())
    mu[b], sig[b] = (X @ f.params)[b], s_[b]
    zs, cw = zfit[k_]
    d.loc[b, "_fold"] = k_
F = lambda zz, k_: np.interp(zz, zfit[k_][0], zfit[k_][1], left=0.0, right=1.0)

eqrows, gaps = [], []
for r in ("AOM", "MBS"):
    m = d[f"{r}_meas"].values; s_rule = d[f"{r}_self"].values; T = TH[r]; dl = d.bmi_self.values - T
    p_el = np.zeros(N)
    for k_ in (0, 1):
        b = d._fold.values == k_
        p_el[b] = 1 - F((T[b] - mu[b]) / sig[b], k_)
    z_uni = (dl > -2) & (dl < 1); share = rate(z_uni, ALLM)
    # 不含種族的機率式：找 p 使總實測比例 ≈ 不對稱規則
    grid = np.linspace(0.001, 0.499, 499)
    meas = np.array([rate((p_el >= p) & (p_el <= 1 - p), ALLM) for p in grid])
    p_star = grid[np.argmin(np.abs(meas - share))]
    z_prob = (p_el >= p_star) & (p_el <= 1 - p_star); fin_prob = np.where(z_prob, m, p_el >= 0.5)
    # 簡單版：尾數 0／5 者 k = 1.5，其他人 k 調到總實測相同
    hp = heap.astype(bool)
    z_h15 = (dl > -3) & (dl < 1.5)
    ks = np.round(np.arange(0.25, 1.5, 0.0125), 4)
    meas_k = np.array([rate(np.where(hp, z_h15, (dl > -2 * kk) & (dl < kk)), ALLM) for kk in ks])
    k_star = ks[np.argmin(np.abs(meas_k - share))]
    z_heap = np.where(hp, z_h15, (dl > -2 * k_star) & (dl < k_star))
    strategies = {"Self-report only": (np.zeros(N, bool), s_rule), "Asymmetric rule, k = 1": (z_uni, s_rule),
                  "Race-free probability, same share measured": (z_prob, fin_prob),
                  "Wider zone for weights ending in 0 or 5, same share measured": (z_heap, s_rule)}
    out_r = {}
    for st, (z, base) in strategies.items():
        fin = np.where(z, m, base); err = m != fin
        out_r[st] = err
        for lab, g in G.items():
            v, bb = rate(err, g), rate(err, g, Wf)
            eqrows.append({"rule": r, "strategy": st, "group": lab, "n": int(g.sum()), "measured": rate(z, g), "residual": v, "lo": pctl(bb)[0], "hi": pctl(bb)[1]})
        eqrows.append({"rule": r, "strategy": st, "group": "All adults", "n": N, "measured": rate(z, ALLM), "residual": rate(err, ALLM),
                       "lo": pctl(rate(err, ALLM, Wf))[0], "hi": pctl(rate(err, ALLM, Wf))[1]})
        wv, wb = rate(err, G["Non-Hispanic White"]), rate(err, G["Non-Hispanic White"], Wf)
        for lab in ("Non-Hispanic Black", "Hispanic", "Non-Hispanic Asian", "Other or multiracial"):
            gv, gb = rate(err, G[lab]), rate(err, G[lab], Wf)
            gaps.append({"rule": r, "strategy": st, "group": lab, "abs_gap_pp": gv - wv, "abs_lo": pctl(gb - wb)[0], "abs_hi": pctl(gb - wb)[1],
                         "ratio": gv / wv, "ratio_lo": pctl(gb / wb)[0], "ratio_hi": pctl(gb / wb)[1]})
    print(r, "share", round(share, 2), "p*", round(p_star, 3), "k*", k_star, flush=True)
pd.DataFrame(eqrows).to_csv(OUT / "r5_equity.csv", index=False)
pd.DataFrame(gaps).to_csv(OUT / "r5_gaps.csv", index=False)

# ---------------------------------------------------------------- 針對性灌水
rng2 = np.random.default_rng(20261024)
U = rng2.random(N)
tg = []
for r, f in RULES.items():
    if r not in ("AOM", "MBS"):
        continue
    m = d[f"{r}_meas"].values; T = TH[r]; b0 = d.bmi_self.values.copy()
    for X_ in (1.0, 2.0, 3.0):
        for p in (0.25, 0.5, 1.0):
            gm = (b0 < T) & (b0 >= T - X_) & (U < p)                      # 有誘因而灌水的人
            bs = np.where(gm, T + 0.1, b0)
            s = f(bs, d); dl = bs - T
            row = {"rule": r, "within_kg_m2": X_, "share_inflating": p, "inflators_pct": rate(gm, ALLM),
                   "discordant": rate(m != s, ALLM), "self_only": rate(~m & s, ALLM), "measured_only": rate(m & ~s, ALLM)}
            for nm, z in [("m2p1", (dl > -2) & (dl < 1)), ("m2p2", (dl > -2) & (dl < 2))]:
                fin = np.where(z, m, s)
                row[f"{nm}_measured"] = rate(z, ALLM); row[f"{nm}_residual"] = rate(m != fin, ALLM); row[f"{nm}_self_only"] = rate(~m & fin, ALLM)
            tg.append(row)
    # 均勻多報 2 kg：盛行率看似正確、個人層級仍誤判
    bs = (d.WHD020.values + 2 * 2.20462) * 703 / d.WHD010.values ** 2; s = f(bs, d)
    tg.append({"rule": r, "within_kg_m2": np.nan, "share_inflating": np.nan, "inflators_pct": 100.0, "discordant": rate(m != s, ALLM),
               "self_only": rate(~m & s, ALLM), "measured_only": rate(m & ~s, ALLM), "uniform_2kg_eligible_measured": rate(m, ALLM), "uniform_2kg_eligible_self": rate(s, ALLM)})
pd.DataFrame(tg).to_csv(OUT / "r5_targeted.csv", index=False)
pd.set_option("display.width", 230)
e = pd.DataFrame(eqrows); print(e.pivot_table(index=["rule", "group"], columns="strategy", values="residual").round(2).to_string())
print(e[e.group == "All adults"][["rule", "strategy", "measured", "residual"]].round(2).to_string())
print(pd.DataFrame(gaps).round(2).to_string()); print(pd.DataFrame(tg).round(2).to_string())
