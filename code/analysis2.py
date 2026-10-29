"""第二輪（外部意見 2026-10-23）：特徵表、分類表現（敏感度／特異度／PPV／NPV／κ、加權人數）、依實測 BMI 標準化的子群比較、
方向拆分、自報 vs 實測散布、校正模型（2013–2018 訓練、2021–2023 時間外驗證）、灰色地帶取捨、亞裔切點等敏感度分析、各週期。

資料準備、資格規則、PSU 重抽權重矩陣 WB 與 est() 直接沿用 analysis.py（執行其前段，避免兩份定義分岔）。
輸出 out/r2_*.csv、out/r2_summary.json。
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
src = (HERE / "analysis.py").read_text()
exec(compile(src[:src.index("t1 = [summary(r)")], "analysis.py", "exec"))   # 定義 d, WB, est, ALL, summary, SUBS, RULES, OUT
N = len(d); w = d.w.values; Wf = WB.astype(np.float64)


def ci_of(bs):
    return np.percentile(bs, [2.5, 97.5])


def wsum(mask):
    """加權人數（百萬）與 CI。"""
    m = mask.astype(np.float64)
    return w @ m / 1e6, *(ci_of(Wf @ m / 1e6))


def wratio(num, den):
    v = 100 * (w @ (num & den)) / (w @ den)
    bs = 100 * (Wf @ (num & den).astype(float)) / (Wf @ den.astype(float))
    return v, *ci_of(bs)


TH = {"AOM": lambda x: np.where(x.com.values, 27.0, 30.0), "MBS": lambda x: np.where(x.met.values, 30.0, 35.0)}
COND = {"AOM": d.com.values, "MBS": d.met.values}

# ---------------------------------------------------------------- Table 1：特徵（全體／一致／不一致，以 AOM 分類）
disc = d.AOM_meas.values != d.AOM_self.values
cols = {"All": np.ones(N, bool), "Concordant": ~disc, "Discordant": disc}
rows = []
def add(label, mask, kind="pct"):
    r = {"characteristic": label}
    for c, cm in cols.items():
        if kind == "pct":
            r[c] = f"{100 * (w @ (mask & cm)) / (w @ cm):.1f} ({int((mask & cm).sum()):,})"
        else:
            v = mask
            r[c] = f"{np.average(v[cm], weights=w[cm]):.1f} ({np.sqrt(np.average((v[cm] - np.average(v[cm], weights=w[cm])) ** 2, weights=w[cm])):.1f})"
    rows.append(r)
rows.append({"characteristic": "Participants, n", **{c: f"{int(cm.sum()):,}" for c, cm in cols.items()}})
rows.append({"characteristic": "Weighted population, millions", **{c: f"{w[cm].sum() / 1e6:.1f}" for c, cm in cols.items()}})
add("Age, years, mean (SD)", d.RIDAGEYR.values.astype(float), "mean")
for lab, m in [("18–39", d.RIDAGEYR < 40), ("40–64", d.RIDAGEYR.between(40, 64)), ("≥ 65", d.RIDAGEYR >= 65)]:
    add(f"Age {lab} years, %", m.values)
add("Women, %", (d.RIAGENDR == 2).values)
for lab, codes in [("Non-Hispanic White", [3]), ("Non-Hispanic Black", [4]), ("Hispanic", [1, 2]), ("Non-Hispanic Asian", [6]), ("Other or multiracial", [7])]:
    add(f"{lab}, %", d.RIDRETH3.isin(codes).values)
add("Measured BMI, kg/m², mean (SD)", d.bmi_meas.values, "mean")
for lab, lo, hi in [("< 25", 0, 25), ("25 to < 27", 25, 27), ("27 to < 30", 27, 30), ("30 to < 35", 30, 35), ("≥ 35", 35, 999)]:
    add(f"Measured BMI {lab} kg/m², %", ((d.bmi_meas >= lo) & (d.bmi_meas < hi)).values)
add("Self-reported minus measured BMI, kg/m², mean (SD)", (d.bmi_self - d.bmi_meas).values, "mean")
add("Any weight-related condition, %", d.com.values)
for lab, m in [("Hypertension", d.BPQ020.eq(1)), ("High cholesterol", d.BPQ080.eq(1)), ("Diabetes", d.DIQ010.eq(1)),
               ("Cardiovascular disease", d[["MCQ160B", "MCQ160C", "MCQ160D", "MCQ160E", "MCQ160F"]].eq(1).any(axis=1))]:
    add(f"   {lab}, %", m.values)
for c in CYCLES:
    add(f"Survey cycle {c.replace('-', '–')}, %", (d.cycle == c).values)
pd.DataFrame(rows).to_csv(OUT / "r2_table1.csv", index=False)

# ---------------------------------------------------------------- Table 2：分類表現（實測為參考）
t2 = []
def kappa(m, s, ww):
    po = ww @ (m == s) / ww.sum()
    pm, ps = ww @ m / ww.sum(), ww @ s / ww.sum()
    pe = pm * ps + (1 - pm) * (1 - ps)
    return (po - pe) / (1 - pe)
for r in ("AOM", "MBS"):
    m, s = d[f"{r}_meas"].values, d[f"{r}_self"].values
    row = {"rule": r}
    for k, num, den in [("sensitivity", s, m), ("specificity", ~s, ~m), ("ppv", m, s), ("npv", ~m, ~s), ("discordant", m != s, np.ones(N, bool)),
                        ("measured_only", m & ~s, np.ones(N, bool)), ("self_only", ~m & s, np.ones(N, bool))]:
        row[k], row[k + "_lo"], row[k + "_hi"] = wratio(num, den)
    kb = [kappa(m, s, Wf[b]) for b in range(REPS)]
    row["kappa"], (row["kappa_lo"], row["kappa_hi"]) = kappa(m, s, w), ci_of(kb)
    for k, mask in [("pop_total", np.ones(N, bool)), ("pop_eligible_measured", m), ("pop_eligible_self", s), ("pop_discordant", m != s),
                    ("pop_measured_only", m & ~s), ("pop_self_only", ~m & s)]:
        row[k], row[k + "_lo"], row[k + "_hi"] = wsum(mask)
    for k, mask in [("n_mm", m & s), ("n_m_only", m & ~s), ("n_s_only", ~m & s), ("n_neither", ~m & ~s)]:
        row[k] = int(mask.sum())
    row["ratio_measured_to_self_only"] = (w @ (m & ~s)) / (w @ (~m & s))
    t2.append(row)
pd.DataFrame(t2).to_csv(OUT / "r2_table2.csv", index=False)
print("tables done", flush=True)

# ---------------------------------------------------------------- 標準化的子群比較（依實測 BMI 1 單位 × 條件）
def strata(r):
    b = np.clip(np.floor(d.bmi_meas.values), 15, 50).astype(int)
    return b * 2 + COND[r].astype(int)
sg = []
for r in ("AOM", "MBS"):
    k = strata(r); ks = np.unique(k); K = len(ks)
    S = (k[:, None] == ks[None, :]).astype(np.float64)                      # n × K
    dis = (d[f"{r}_meas"].values != d[f"{r}_self"].values).astype(float)
    p_all = (w @ S) / w.sum(); p_all_b = (Wf @ S) / Wf.sum(1, keepdims=True)
    for grp, lab, fn in [("All", "All", ALL)] + SUBS:
        g = fn(d).astype(float)
        den = w @ (S * g[:, None]); num = w @ (S * (g * dis)[:, None])
        ok = den > 0
        rate = np.where(ok, num / np.where(ok, den, 1), 0)
        std = 100 * (p_all[ok] @ rate[ok]) / p_all[ok].sum()
        denb = Wf @ (S * g[:, None]); numb = Wf @ (S * (g * dis)[:, None])
        rateb = np.where(denb > 0, numb / np.where(denb > 0, denb, 1), 0)
        stdb = 100 * ((p_all_b[:, ok] * rateb[:, ok]).sum(1)) / p_all_b[:, ok].sum(1)
        m, s = d[f"{r}_meas"].values, d[f"{r}_self"].values; gb = g.astype(bool)
        cr = wratio(m != s, gb); mo = wratio(m & ~s, gb); so = wratio(~m & s, gb)
        # 門檻 ±1.5 kg/m² 內的人占比：說明粗率差異有多少來自「門檻附近的人多寡」
        near = np.abs(d.bmi_meas.values - TH[r](d)) < 1.5
        sg.append({"rule": r, "group": grp, "subgroup": lab, "n": int(gb.sum()), "crude": cr[0], "crude_lo": cr[1], "crude_hi": cr[2],
                   "standardized": std, "std_lo": np.percentile(stdb, 2.5), "std_hi": np.percentile(stdb, 97.5),
                   "measured_only": mo[0], "self_only": so[0], "near_threshold_pct": 100 * (w @ (near & gb)) / (w @ gb),
                   "strata_used_pct_of_population": 100 * p_all[ok].sum()})
pd.DataFrame(sg).to_csv(OUT / "r2_subgroups.csv", index=False)
print("standardized subgroups done", flush=True)

# ---------------------------------------------------------------- 依實測 BMI 區間：P(自報判合格)、方向拆分、人口密度
bins = np.arange(18, 45.01, 0.5); bi = np.digitize(d.bmi_meas.values, bins) - 1
cur = []
for r in ("AOM", "MBS"):
    m, s = d[f"{r}_meas"].values, d[f"{r}_self"].values
    for cl, cm in [("with condition", COND[r]), ("without condition", ~COND[r]), ("all", np.ones(N, bool))]:
        for j in range(len(bins) - 1):
            g = (bi == j) & cm
            if g.sum() < 10:
                continue
            gf = g.astype(float)
            pe = 100 * (w @ (gf * s)) / (w @ gf); peb = 100 * (Wf @ (gf * s)) / (Wf @ gf)
            cur.append({"rule": r, "condition": cl, "bmi_mid": bins[j] + 0.25, "n": int(g.sum()),
                        "p_self_eligible": pe, "p_lo": np.percentile(peb, 2.5), "p_hi": np.percentile(peb, 97.5),
                        "p_measured_eligible": 100 * (w @ (gf * m)) / (w @ gf),
                        "measured_only": 100 * (w @ (gf * (m & ~s))) / (w @ gf), "self_only": 100 * (w @ (gf * (~m & s))) / (w @ gf),
                        "pop_share": 100 * (w @ gf) / w.sum()})
pd.DataFrame(cur).to_csv(OUT / "r2_curves.csv", index=False)

# 散布圖象限（只以無條件時的門檻 30／35 劃線；百分比為全體加權）
quad = {}
for r, T in (("AOM", 30), ("MBS", 35)):
    a, b_ = d.bmi_meas.values >= T, d.bmi_self.values >= T
    quad[r] = {"threshold": T, "meas_ge_self_lt": 100 * (w @ (a & ~b_)) / w.sum(), "meas_lt_self_ge": 100 * (w @ (~a & b_)) / w.sum(),
               "both_ge": 100 * (w @ (a & b_)) / w.sum(), "both_lt": 100 * (w @ (~a & ~b_)) / w.sum()}
d[["bmi_meas", "bmi_self", "w"]].to_csv(OUT / "r2_scatter.csv", index=False)

# ---------------------------------------------------------------- 校正模型：2013–2018 訓練、2021–2023 驗證
import statsmodels.api as sm
X = pd.DataFrame({"self_bmi": d.bmi_self, "self_bmi_sq": (d.bmi_self - 30) ** 2, "age10": (d.RIDAGEYR - 50) / 10,
                  "women": (d.RIAGENDR == 2).astype(float), "black": (d.RIDRETH3 == 4).astype(float),
                  "hispanic": d.RIDRETH3.isin([1, 2]).astype(float), "asian": (d.RIDRETH3 == 6).astype(float),
                  "other_race": (d.RIDRETH3 == 7).astype(float)})
X["women_x_self"] = X.women * (X.self_bmi - 30); X = sm.add_constant(X)
tr, te = (d.cycle != "2021-2023").values, (d.cycle == "2021-2023").values
fit = sm.WLS(d.bmi_meas.values[tr], X.values[tr], weights=w[tr]).fit()
pred = X.values @ fit.params
d["bmi_corr"] = pred
for r, f in RULES.items():
    d[f"{r}_corr"] = f(pred, d)
coef = pd.DataFrame({"term": X.columns, "coef": fit.params, "se": fit.bse, "lo": fit.params - 1.96 * fit.bse, "hi": fit.params + 1.96 * fit.bse})
perf = []
for lab, mask in [("training 2013–2018", tr), ("validation 2021–2023", te)]:
    e_raw = d.bmi_self.values[mask] - d.bmi_meas.values[mask]; e_cor = pred[mask] - d.bmi_meas.values[mask]
    perf.append({"set": lab, "n": int(mask.sum()), "mean_error_self": np.average(e_raw, weights=w[mask]), "mean_error_corrected": np.average(e_cor, weights=w[mask]),
                 "mae_self": np.average(np.abs(e_raw), weights=w[mask]), "mae_corrected": np.average(np.abs(e_cor), weights=w[mask])})
cal = []
for r in ("AOM", "MBS"):
    m = d[f"{r}_meas"].values
    for lab, col in [("Self-reported", f"{r}_self"), ("Corrected self-reported", f"{r}_corr")]:
        v = wratio(m != d[col].values, te); mo = wratio(m & ~d[col].values, te); so = wratio(~m & d[col].values, te)
        cal.append({"rule": r, "input": lab, "set": "validation 2021–2023", "discordant": v[0], "lo": v[1], "hi": v[2],
                    "measured_only": mo[0], "self_only": so[0]})
pd.DataFrame(cal).to_csv(OUT / "r2_calibration.csv", index=False)
coef.to_csv(OUT / "r2_model_coef.csv", index=False); pd.DataFrame(perf).to_csv(OUT / "r2_model_perf.csv", index=False)

# ---------------------------------------------------------------- 灰色地帶：自報（或校正後）BMI 距門檻 < k 時改用實測
gz = []
for r in ("AOM", "MBS"):
    m = d[f"{r}_meas"].values; T = TH[r](d)
    for inp, b, mask in [("Self-reported, all cycles", d.bmi_self.values, np.ones(N, bool)),
                         ("Corrected self-reported, validation 2021–2023", pred, te)]:
        s_rule = d[f"{r}_self"].values if inp.startswith("Self") else d[f"{r}_corr"].values
        for k in np.round(np.arange(0, 4.01, 0.25), 2):
            zone = np.abs(b - T) < k
            final = np.where(zone, m, s_rule)
            v = wratio(m != final, mask); z = wratio(zone, mask)
            gz.append({"rule": r, "input": inp, "k": k, "measured_pct": z[0], "measured_lo": z[1], "measured_hi": z[2],
                       "residual_discordant": v[0], "res_lo": v[1], "res_hi": v[2]})
pd.DataFrame(gz).to_csv(OUT / "r2_greyzone.csv", index=False)
print("calibration + grey zone done", flush=True)

# ---------------------------------------------------------------- 敏感度分析（追加）
asian = (d.RIDRETH3 == 6).values
sens2 = []
def s2(name, rule, m, s, mask=np.ones(N, bool)):
    v = wratio(m != s, mask); sens2.append({"analysis": name, "rule": rule, "n": int(mask.sum()), "discordant": v[0], "lo": v[1], "hi": v[2]})
prev = pd.read_csv(OUT / "sensitivity.csv")
# 亞裔切點：AOM 照 NICE TA875「門檻通常降 2.5 kg/m²」套到仿單門檻 → ≥ 27.5，或 ≥ 24.5 且有條件；MBS ≥ 27.5（ASMBS／IFSO 2022）
aom_as = lambda b: (b >= 27.5) | ((b >= 24.5) & d.com.values)
mbs_as = lambda b: b >= 27.5
s2("Asian-specific thresholds, non-Hispanic Asian adults", "AOM", aom_as(d.bmi_meas.values), aom_as(d.bmi_self.values), asian)
s2("Asian-specific thresholds, non-Hispanic Asian adults", "MBS", mbs_as(d.bmi_meas.values), mbs_as(d.bmi_self.values), asian)
s2("Standard thresholds, non-Hispanic Asian adults (for comparison)", "AOM", d.AOM_meas.values, d.AOM_self.values, asian)
s2("Standard thresholds, non-Hispanic Asian adults (for comparison)", "MBS", d.MBS_meas.values, d.MBS_self.values, asian)
ext = d.bmi_meas.between(18.5, 60).values & d.bmi_self.between(18.5, 60).values
for r in ("AOM", "MBS"):
    s2("Excluding measured or self-reported BMI < 18.5 or > 60 kg/m²", r, d[f"{r}_meas"].values, d[f"{r}_self"].values, ext)
com2 = (d.BPQ020.eq(1) | d.DIQ010.eq(1) | d[["MCQ160B", "MCQ160C", "MCQ160D", "MCQ160E", "MCQ160F"]].eq(1).any(axis=1)).values
aom2 = lambda b: (b >= 30) | ((b >= 27) & com2)
s2("Weight-related condition without high cholesterol", "AOM", aom2(d.bmi_meas.values), aom2(d.bmi_self.values))
pd.concat([prev, pd.DataFrame(sens2)]).to_csv(OUT / "r2_sensitivity.csv", index=False)

# ---------------------------------------------------------------- 各週期
cyc = []
for r in ("AOM", "MBS"):
    m, s = d[f"{r}_meas"].values, d[f"{r}_self"].values
    for c in CYCLES:
        cm = (d.cycle == c).values; v = wratio(m != s, cm); mo = wratio(m & ~s, cm)
        cyc.append({"rule": r, "cycle": c, "n": int(cm.sum()), "discordant": v[0], "lo": v[1], "hi": v[2], "measured_only": mo[0]})
pd.DataFrame(cyc).to_csv(OUT / "r2_cycles.csv", index=False)

# 末位數（加權）
dig = pd.DataFrame({"digit": np.arange(10),
                    "weight_pct": [100 * (w @ (d.WHD020.values % 10 == k)) / w.sum() for k in range(10)],
                    "height_pct": [100 * (w @ (d.WHD010.values % 10 == k)) / w.sum() for k in range(10)]})
dig.to_csv(OUT / "r2_digits.csv", index=False)
json.dump({"quadrants": quad, "model_r2": fit.rsquared, "reps": REPS}, open(OUT / "r2_summary.json", "w"), indent=1)
print(pd.DataFrame(t2)[["rule", "sensitivity", "specificity", "ppv", "npv", "kappa", "pop_discordant", "pop_discordant_lo", "pop_discordant_hi", "pop_measured_only", "ratio_measured_to_self_only"]].round(3).to_string())
print(pd.DataFrame(cal).round(2).to_string())
print(pd.DataFrame(sg).query("group in ['All','Race and ethnicity']")[["rule", "subgroup", "crude", "standardized", "std_lo", "std_hi", "near_threshold_pct"]].round(2).to_string())
