"""第四輪（外部意見 2026-10-23）：
1. 人數 CI：每個 bootstrap replicate 的權重在各週期內重新校正（re-rake）到該週期的原始權重總和，人數＝率×固定人口，CI 與率一致。
2. 實測策略（2021–2023 驗證）：對稱、不對稱（下:上＝1.5、2、3）、機率式（改用訓練期標準化殘差的經驗分位數，不假設常態）；
   在相同實測比例下以配對 bootstrap 比較殘餘誤判。
3. 公平性：不對稱規則（k = 1）下各族群的殘餘誤判、實測比例，以及各族群達到 ≤ 2% 所需的 k。
4. What-if：自報體重系統性多報 1、2、3 kg（遠距開藥有誘因時），只有自報合格（不該合格卻合格）的比例與不對稱規則下的殘餘。

資料、規則與 WB 沿用 analysis.py 前段；模型同 analysis3.py。輸出 out/r4_*.csv、out/r4_summary.json。
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
pctl = lambda bs: np.percentile(bs, [2.5, 97.5], axis=0)
TH = {"AOM": np.where(d.com.values, 27.0, 30.0), "MBS": np.where(d.met.values, 30.0, 35.0)}
tr, te = (d.cycle != "2021-2023").values, (d.cycle == "2021-2023").values
ALLM = np.ones(N, bool)
out = {}

# ---------------------------------------------------------------- 1. re-rake
cyc = d.cycle.values
Wr = Wf.copy()
for c in np.unique(cyc):
    m = cyc == c
    Wr[:, m] *= (w[m].sum() / Wr[:, m].sum(1))[:, None]


def rate(num, den, W=None):
    num = num & den
    if W is None:
        return 100 * (w @ num) / (w @ den)
    return 100 * (W @ num.astype(float)) / (W @ den.astype(float))


pop = []
for r in ("AOM", "MBS"):
    m, s = d[f"{r}_meas"].values, d[f"{r}_self"].values
    for scope, mask, f in [("all cycles (average)", ALLM, 1.0), ("2021–2023 cycle", te, 4.0)]:
        for k, x in [("total", ALLM), ("eligible_measured", m), ("eligible_self", s), ("discordant", m != s), ("measured_only", m & ~s), ("self_only", ~m & s)]:
            v = f * (w @ (x & mask)) / 1e6; b = f * (Wr @ (x & mask).astype(float)) / 1e6
            rt = rate(x, mask); rb = rate(x, mask, Wr)
            pop.append({"rule": r, "scope": scope, "quantity": k, "millions": v, "lo": pctl(b)[0], "hi": pctl(b)[1],
                        "rate": rt, "rate_lo": pctl(rb)[0], "rate_hi": pctl(rb)[1]})
pop = pd.DataFrame(pop); pop.to_csv(OUT / "r4_population.csv", index=False)

# ---------------------------------------------------------------- 2. 模型（同 analysis3）＋ 經驗殘差分佈
heap = np.isin(d.WHD020.values % 10, [0, 5]).astype(float)
X = np.column_stack([np.ones(N), d.bmi_self, (d.RIDAGEYR - 50) / 10, (d.RIAGENDR == 2), d.RIDRETH3 == 4, d.RIDRETH3.isin([1, 2]),
                     d.RIDRETH3 == 6, d.RIDRETH3 == 7, heap]).astype(float)
fit = sm.WLS(d.bmi_meas.values[tr], X[tr], weights=w[tr]).fit(); mu = X @ fit.params; res = d.bmi_meas.values - mu
vfit = sm.WLS(np.log(res[tr] ** 2 + 1e-6), X[tr], weights=w[tr]).fit(); lv = X @ vfit.params
sigma = np.sqrt(np.exp(lv) * np.average(res[tr] ** 2, weights=w[tr]) / np.average(np.exp(lv[tr]), weights=w[tr]))
z_tr = res[tr] / sigma[tr]; o = np.argsort(z_tr); zs, cw = z_tr[o], np.cumsum(w[tr][o]) / w[tr].sum()
F = lambda z: np.interp(z, zs, cw, left=0.0, right=1.0)                       # 加權經驗 CDF
zq = lambda q: np.interp(q, cw, zs)
z_te = res[te] / sigma[te]
cov = {}
for q in (0.5, 0.8, 0.95):
    lo, hi = zq(0.5 - q / 2), zq(0.5 + q / 2)
    cov[f"pi{int(q * 100)}_empirical"] = float(np.average((z_te > lo) & (z_te < hi), weights=w[te]))
out["coverage_validation_empirical"] = cov

# ---------------------------------------------------------------- 實測策略前緣（含 replicate）
STRATS = {}
def frontier(r):
    m = d[f"{r}_meas"].values; s_rule = d[f"{r}_self"].values; T = TH[r]; dl = d.bmi_self.values - T
    p_el = 1 - F((T - mu) / sigma)
    defs = [("Symmetric ± k", np.round(np.arange(0, 4.01, 0.125), 3), lambda k: (dl > -k) & (dl < k), s_rule),
            ("Asymmetric 1.5:1", np.round(np.arange(0, 3.01, 0.125), 3), lambda k: (dl > -1.5 * k) & (dl < k), s_rule),
            ("Asymmetric 2:1", np.round(np.arange(0, 3.01, 0.125), 3), lambda k: (dl > -2 * k) & (dl < k), s_rule),
            ("Asymmetric 3:1", np.round(np.arange(0, 3.01, 0.125), 3), lambda k: (dl > -3 * k) & (dl < k), s_rule),
            ("Probability-based (empirical)", [0.5, 0.45, 0.4, 0.35, 0.3, 0.25, 0.2, 0.15, 0.1, 0.075, 0.05, 0.035, 0.025, 0.015, 0.01, 0.005, 0.0025, 0.001],
             lambda p: (p_el >= p) & (p_el <= 1 - p) if p < 0.5 else np.zeros(N, bool), p_el >= 0.5)]
    rows, reps = [], {}
    for name, grid, zf, base in defs:
        MB, RB = [], []
        for v in grid:
            z = zf(v); final = np.where(z, m, base)
            mb, rb = rate(z, te, Wr), rate(m != final, te, Wr); MB.append(mb); RB.append(rb)
            rows.append({"rule": r, "strategy": name, "value": v, "measured_pct": rate(z, te), "measured_lo": pctl(mb)[0], "measured_hi": pctl(mb)[1],
                         "residual": rate(m != final, te), "res_lo": pctl(rb)[0], "res_hi": pctl(rb)[1],
                         "res_measured_only": rate(m & ~final, te), "res_self_only": rate(~m & final, te)})
        reps[name] = (np.array(MB), np.array(RB))      # (grid, REPS)
    return pd.DataFrame(rows), reps
FR, CMP = [], []
for r in ("AOM", "MBS"):
    f, reps = frontier(r); FR.append(f)
    ref = f[(f.strategy == "Asymmetric 2:1") & np.isclose(f.value, 1.0)].iloc[0]
    target = ref.measured_pct
    def at_share(name, b=None):
        MB, RB = reps[name]
        if b is None:
            g = f[f.strategy == name].sort_values("measured_pct"); return np.interp(target, g.measured_pct, g.residual)
        o_ = np.argsort(MB[:, b]); return np.interp(target, MB[o_, b], RB[o_, b])
    base_b = np.array([at_share("Asymmetric 2:1", b) for b in range(REPS)])
    for name in reps:
        if name == "Asymmetric 2:1":
            continue
        vb = np.array([at_share(name, b) for b in range(REPS)])
        dlt = vb - base_b
        CMP.append({"rule": r, "strategy": name, "measured_share": target, "residual_at_share": at_share(name),
                    "residual_asym2": at_share("Asymmetric 2:1"), "difference_vs_asym2": at_share(name) - at_share("Asymmetric 2:1"),
                    "diff_lo": pctl(dlt)[0], "diff_hi": pctl(dlt)[1]})
fr = pd.concat(FR); fr.to_csv(OUT / "r4_frontier.csv", index=False); pd.DataFrame(CMP).to_csv(OUT / "r4_paired.csv", index=False)
ops = []
for (r, st), g in fr.groupby(["rule", "strategy"]):
    g = g.sort_values("measured_pct")
    for tg in (3, 2, 1):
        ok = g[g.residual <= tg]
        if len(ok):
            ops.append({"rule": r, "strategy": st, "target": tg, "measured_pct": ok.measured_pct.iloc[0], "value": ok.value.iloc[0], "residual": ok.residual.iloc[0]})
pd.DataFrame(ops).to_csv(OUT / "r4_operating_points.csv", index=False)
print("frontier done", flush=True)

# ---------------------------------------------------------------- 3. 公平性（全部週期；不對稱 2:1 規則不需模型）
RACE = [("Non-Hispanic White", [3]), ("Non-Hispanic Black", [4]), ("Hispanic", [1, 2]), ("Non-Hispanic Asian", [6]), ("Other or multiracial", [7]), ("All adults", [1, 2, 3, 4, 6, 7])]
eq = []
for r in ("AOM", "MBS"):
    m = d[f"{r}_meas"].values; s_rule = d[f"{r}_self"].values; dl = d.bmi_self.values - TH[r]
    for lab, codes in RACE:
        g = np.isin(d.RIDRETH3.values, codes)
        row = {"rule": r, "group": lab, "n": int(g.sum())}
        for k_, nm in [(0.0, "none"), (1.0, "k1")]:
            z = (dl > -2 * k_) & (dl < k_); final = np.where(z, m, s_rule)
            v, b = rate(m != final, g), rate(m != final, g, Wr)
            row[f"residual_{nm}"], row[f"res_{nm}_lo"], row[f"res_{nm}_hi"] = v, *pctl(b)
            row[f"measured_{nm}"] = rate(z, g)
        need = None
        for k_ in np.round(np.arange(0, 3.01, 0.125), 3):
            z = (dl > -2 * k_) & (dl < k_)
            if rate(m != np.where(z, m, s_rule), g) <= 2:
                need = (k_, rate(z, g)); break
        row["k_for_2pct"], row["measured_for_2pct"] = (need if need else (np.nan, np.nan))
        eq.append(row)
pd.DataFrame(eq).to_csv(OUT / "r4_equity.csv", index=False)
print("equity done", flush=True)

# ---------------------------------------------------------------- 4. What-if：自報體重多報 Δ kg
wi = []
for r, f in RULES.items():
    if r not in ("AOM", "MBS"):
        continue
    m = d[f"{r}_meas"].values; T = TH[r]
    for dk in (0, 1, 2, 3):
        bs = (d.WHD020.values + dk * 2.20462) * 703 / d.WHD010.values ** 2
        s = f(bs, d); dl = bs - T
        z = (dl > -2) & (dl < 1); final = np.where(z, m, s)
        z2 = (dl > -2) & (dl < 2); final2 = np.where(z2, m, s)
        wi.append({"rule": r, "added_kg": dk, "discordant": rate(m != s, ALLM), "measured_only": rate(m & ~s, ALLM), "self_only": rate(~m & s, ALLM),
                   "asym_k1_measured": rate(z, ALLM), "asym_k1_residual": rate(m != final, ALLM), "asym_k1_self_only": rate(~m & final, ALLM),
                   "sym_up2_measured": rate(z2, ALLM), "sym_up2_residual": rate(m != final2, ALLM), "sym_up2_self_only": rate(~m & final2, ALLM)})
pd.DataFrame(wi).to_csv(OUT / "r4_whatif.csv", index=False)
json.dump(out, open(OUT / "r4_summary.json", "w"), indent=1)
pd.set_option("display.width", 220)
print(json.dumps(out, indent=1))
print(pop[pop.quantity == "discordant"].round(2).to_string())
print(pd.DataFrame(CMP).round(2).to_string()); print(pd.DataFrame(ops).round(2).to_string())
print(pd.DataFrame(eq).round(2).to_string()); print(pd.DataFrame(wi).round(2).to_string())
