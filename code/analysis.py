"""正式分析：減重藥（AOM）與減重手術（MBS）資格，自報 vs 實測身高體重的判定不一致。

事前目標（寫於讀正式結果前，2026-10-23；試跑見 bmi_pilot2.py）：估計以自報身高體重判定的資格與以實測值判定者不一致的比例
（主要量），並描述方向、在 BMI 門檻附近的集中程度與子群差異。分析單位：受訪成人（每人一筆）。
設計：NHANES 2013–2014、2015–2016、2017–2018、2021–2023（MEC 權重 /4），信賴區間以層內 PSU 重抽 1,000 次（percentile）。
資格規則（仿單／指引；稿件前查原文）：AOM＝BMI ≥ 30，或 ≥ 27 且至少一個體重相關共病（自述高血壓、高膽固醇、糖尿病、心血管病）；
MBS（ASMBS／IFSO 2022）＝BMI ≥ 35，或 ≥ 30 且代謝病（自述糖尿病或高血壓）。

輸出 out/*.csv、out/flow.json。
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DAT, OUT = ROOT / "data", ROOT / "out"
OUT.mkdir(exist_ok=True)
rng = np.random.default_rng(20261023)
REPS = 1000
CYCLES = ["2013-2014", "2015-2016", "2017-2018", "2021-2023"]
WHQ = {"2013-2014": "H", "2015-2016": "I", "2017-2018": "J", "2021-2023": "L"}


def rd(p, cols):
    d = pd.read_sas(p)
    return d[[c for c in cols if c in d.columns]]


def load(cyc):
    f = (lambda c: DAT / f"{c}_L.xpt") if cyc == "2021-2023" else (lambda c: DAT / f"{c}_{WHQ[cyc]}.xpt")
    d = rd(f("DEMO"), ["SEQN", "RIAGENDR", "RIDAGEYR", "RIDRETH3", "RIDEXPRG", "WTMEC2YR", "SDMVPSU", "SDMVSTRA", "RIDSTATR"])
    d = d.merge(rd(f("BMX"), ["SEQN", "BMXWT", "BMIWT", "BMXHT", "BMIHT"]), on="SEQN", how="left")
    d = d.merge(rd(DAT / f"WHQ_{WHQ[cyc]}.xpt", ["SEQN", "WHD010", "WHD020"]), on="SEQN", how="left")
    d = d.merge(rd(f("BPQ"), ["SEQN", "BPQ020", "BPQ080"]), on="SEQN", how="left")
    d = d.merge(rd(f("DIQ"), ["SEQN", "DIQ010"]), on="SEQN", how="left")
    d = d.merge(rd(f("MCQ"), ["SEQN", "MCQ160B", "MCQ160C", "MCQ160D", "MCQ160E", "MCQ160F"]), on="SEQN", how="left")
    d["cycle"] = cyc
    return d


d = pd.concat([load(c) for c in CYCLES], ignore_index=True)
flow = {"examined (all ages)": int((d.RIDSTATR == 2).sum())}
d = d[d.RIDSTATR == 2]
d = d[d.RIDAGEYR >= 18]; flow["adults >= 18 examined"] = len(d)
preg = d.RIDEXPRG.eq(1); flow["excluded: pregnant"] = int(preg.sum()); d = d[~preg]
nomeas = d.BMXWT.isna() | d.BMXHT.isna(); flow["excluded: measured height or weight missing"] = int(nomeas.sum()); d = d[~nomeas]
flag = d.BMIWT.notna() | d.BMIHT.notna(); flow["excluded: measurement comment (clothing, appliance, other)"] = int(flag.sum()); d = d[~flag]
noself = ~(d.WHD010.between(48, 90) & d.WHD020.between(75, 600))
flow["excluded: self-reported height or weight missing, refused, don't know or implausible"] = int(noself.sum()); d = d[~noself]
d = d[d.WTMEC2YR > 0]
flow["analysed"] = len(d)
flow["analysed by cycle"] = d.cycle.value_counts().sort_index().to_dict()
json.dump(flow, open(OUT / "flow.json", "w"), indent=1)

d["w"] = d.WTMEC2YR / 4
d["stratum"] = d.cycle + "_" + d.SDMVSTRA.astype(int).astype(str)
d["psu"] = d.stratum + "_" + d.SDMVPSU.astype(int).astype(str)
d["bmi_self"] = d.WHD020 * 703 / d.WHD010 ** 2
d["bmi_meas"] = d.BMXWT / (d.BMXHT / 100) ** 2
d["bmi_round"] = np.round(d.BMXWT * 2.20462) * 703 / np.round(d.BMXHT / 2.54) ** 2
d["com"] = (d.BPQ020.eq(1) | d.BPQ080.eq(1) | d.DIQ010.eq(1) | d[["MCQ160B", "MCQ160C", "MCQ160D", "MCQ160E", "MCQ160F"]].eq(1).any(axis=1))
d["met"] = d.DIQ010.eq(1) | d.BPQ020.eq(1)
d = d.reset_index(drop=True)

RULES = {"AOM": lambda b, x: (b >= 30) | ((b >= 27) & x.com.values),
         "MBS": lambda b, x: (b >= 35) | ((b >= 30) & x.met.values),
         "BMI >= 30 only": lambda b, x: b >= 30}
for k, f in RULES.items():
    for s in ("meas", "self", "round"):
        d[f"{k}_{s}"] = f(d[f"bmi_{s}"].values, d)

# ---------------------------------------------------------------- bootstrap（層內 PSU 重抽）
# Rao–Wu rescaling bootstrap：每層 n_h 個 PSU 中有放回抽 m_h = n_h − 1 個，被抽中的 PSU 權重乘 n_h/(n_h − 1) × 次數
# （2026-10-23 審閱指出：原本每層抽 n_h = 2 個且不調整，變異數低估一半、CI 約窄 √2 倍；已改）
# 計數矩陣 REPS × PSU，估計值＝加權和的比值，避免逐次複製資料
psu_codes, psu_idx = np.unique(d.psu.values, return_inverse=True)
str_of_psu = pd.Series(d.stratum.values).groupby(psu_idx).first().values
by_stratum = {s_: np.where(str_of_psu == s_)[0] for s_ in np.unique(str_of_psu)}
C = np.zeros((REPS, len(psu_codes)), dtype=np.float32)
for b in range(REPS):
    for s_, ps in by_stratum.items():
        nh = len(ps)
        np.add.at(C[b], rng.choice(ps, size=nh - 1, replace=True), nh / (nh - 1))
WB = C[:, psu_idx] * d.w.values.astype(np.float32)   # REPS × n 的重抽權重
# 每個 replicate 在週期內重新校正（re-rake）到該週期的原始權重總和：NHANES 權重已校正到普查人口，人口總數不應隨重抽變動
for _c in d.cycle.unique():
    _m = (d.cycle == _c).values
    WB[:, _m] *= (d.w.values[_m].sum() / WB[:, _m].sum(1))[:, None].astype(np.float32)


def est(mask_fn, sub):
    """mask_fn(df) -> (numerator bool, denominator bool)；回傳加權 % 與 bootstrap 95% CI。"""
    num, den = mask_fn(d)
    den = den & sub(d)
    v = 100 * d.w.values[num & den].sum() / d.w.values[den].sum()
    bs = 100 * (WB @ (num & den).astype(np.float32)) / (WB @ den.astype(np.float32))
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return v, lo, hi


ALL = lambda x: np.ones(len(x), bool)


def summary(rule, sub=ALL, label="All", group="All"):
    m, s = f"{rule}_meas", f"{rule}_self"
    x = d[sub(d)]
    row = {"rule": rule, "group": group, "subgroup": label, "n": len(x)}
    for k, fn in [("eligible_measured", lambda y: (y[m].values, ALL(y))),
                  ("eligible_self", lambda y: (y[s].values, ALL(y))),
                  ("discordant", lambda y: (y[m].values != y[s].values, ALL(y))),
                  ("measured_only", lambda y: (y[m].values & ~y[s].values, ALL(y))),
                  ("self_only", lambda y: (~y[m].values & y[s].values, ALL(y)))]:
        row[k], row[k + "_lo"], row[k + "_hi"] = est(fn, sub)
    row["n_discordant"] = int((x[m] != x[s]).sum())
    return row


SUBS = [("Sex", "Women", lambda x: x.RIAGENDR.values == 2), ("Sex", "Men", lambda x: x.RIAGENDR.values == 1),
        ("Age, years", "18–39", lambda x: x.RIDAGEYR.values < 40), ("Age, years", "40–64", lambda x: (x.RIDAGEYR.values >= 40) & (x.RIDAGEYR.values < 65)),
        ("Age, years", "≥ 65", lambda x: x.RIDAGEYR.values >= 65),
        ("Race and ethnicity", "Non-Hispanic White", lambda x: x.RIDRETH3.values == 3), ("Race and ethnicity", "Non-Hispanic Black", lambda x: x.RIDRETH3.values == 4),
        ("Race and ethnicity", "Hispanic", lambda x: np.isin(x.RIDRETH3.values, [1, 2])), ("Race and ethnicity", "Non-Hispanic Asian", lambda x: x.RIDRETH3.values == 6),
        ("Race and ethnicity", "Other or multiracial", lambda x: x.RIDRETH3.values == 7),
        ("Weight-related condition", "Any", lambda x: x.com.values), ("Weight-related condition", "None", lambda x: ~x.com.values)] + \
       [("Survey cycle", c, (lambda c: lambda x: x.cycle.values == c)(c)) for c in CYCLES]

t1 = [summary(r) for r in ("AOM", "MBS")]
pd.DataFrame(t1).to_csv(OUT / "table1.csv", index=False)
sg = [summary(r, f, lab, grp) for r in ("AOM", "MBS") for grp, lab, f in SUBS]
pd.DataFrame(sg).to_csv(OUT / "subgroups.csv", index=False)
print("table1 + subgroups done", flush=True)

# 敏感度分析
sens = []
def srow(name, rule, a, b, sub=ALL):
    v, lo, hi = est(lambda y: (y[a].values != y[b].values, ALL(y)), sub)
    sens.append({"analysis": name, "rule": rule, "n": int(sub(d).sum()), "discordant": v, "lo": lo, "hi": hi})
for r in ("AOM", "MBS"):
    srow("Primary: self-reported vs measured", r, f"{r}_meas", f"{r}_self")
    srow("Measured values rounded to whole inches and pounds vs measured (rounding alone)", r, f"{r}_meas", f"{r}_round")
    srow("Self-reported vs measured values rounded to whole inches and pounds", r, f"{r}_round", f"{r}_self")
    srow("2021–2023 cycle only", r, f"{r}_meas", f"{r}_self", lambda x: x.cycle.values == "2021-2023")
    srow("Self-reported weight not ending in 0 or 5", r, f"{r}_meas", f"{r}_self", lambda x: ~np.isin(x.WHD020.values % 10, [0, 5]))
srow("BMI criterion alone (≥ 30), comorbidity ignored", "BMI >= 30 only", "BMI >= 30 only_meas", "BMI >= 30 only_self")
pd.DataFrame(sens).to_csv(OUT / "sensitivity.csv", index=False)
print("sensitivity done", flush=True)

# 圖 1A：依實測 BMI 區間的不一致比例（點估計＋Wilson 不加權 n）
bins = np.arange(22, 42.01, 0.5)
d["bin"] = pd.cut(d.bmi_meas, bins, right=False)
rows = []
for r in ("AOM", "MBS"):
    for com_lab, cm in [("with condition", d.com if r == "AOM" else d.met), ("without condition", ~(d.com if r == "AOM" else d.met))]:
        for b, g in d[cm.values].groupby("bin", observed=True):
            if len(g) < 10:
                continue
            dis = (g[f"{r}_meas"] != g[f"{r}_self"])
            rows.append({"rule": r, "condition": com_lab, "bmi_low": b.left, "bmi_mid": b.left + 0.25, "n": len(g),
                         "discordant_pct": 100 * g.w[dis].sum() / g.w.sum()})
pd.DataFrame(rows).to_csv(OUT / "by_bmi_bin.csv", index=False)

# 補充圖：差值分布與尾數
diff = pd.DataFrame({"weight_diff_kg": d.WHD020 / 2.20462 - d.BMXWT, "height_diff_cm": d.WHD010 * 2.54 - d.BMXHT,
                     "bmi_diff": d.bmi_self - d.bmi_meas, "w": d.w, "sex": np.where(d.RIAGENDR == 2, "Women", "Men")})
diff.to_csv(OUT / "differences.csv", index=False)
dig = pd.DataFrame({"digit": np.arange(10),
                    "weight_pct": [100 * (d.WHD020 % 10 == k).mean() for k in range(10)],
                    "height_pct": [100 * (d.WHD010 % 12 == k).mean() for k in range(10)]})
dig.to_csv(OUT / "terminal_digits.csv", index=False)
summ = {"n": len(d), "mean_bmi_diff_weighted": float(np.average(d.bmi_self - d.bmi_meas, weights=d.w)),
        "mean_weight_diff_kg": float(np.average(diff.weight_diff_kg, weights=d.w)),
        "mean_height_diff_cm": float(np.average(diff.height_diff_cm, weights=d.w)),
        "weight_end_0_5_pct": float(100 * np.isin(d.WHD020 % 10, [0, 5]).mean()),
        "com_pct": float(100 * np.average(d.com, weights=d.w)), "reps": REPS}
json.dump(summ, open(OUT / "summary.json", "w"), indent=1)
print(json.dumps(summ, indent=1))
print(pd.DataFrame(t1)[["rule", "n", "eligible_measured", "eligible_self", "discordant", "discordant_lo", "discordant_hi", "measured_only", "self_only"]].round(2).to_string())
