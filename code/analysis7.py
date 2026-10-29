"""第八輪（交叉審閱 2026-10-23）追加的敏感度分析與變異數檢查。

1. 手術的代謝疾病只用糖尿病（ASMBS／IFSO 2022 未定義 metabolic disease；高血壓算進去是本研究的詮釋）。
2. 條件改用「自報或實測」（仿 Wong 2025：血壓平均 ≥ 130/80 mmHg、總膽固醇 ≥ 200 mg/dL；糖尿病另加 HbA1c ≥ 6.5%）。
   2021–2023 的血壓是示波法（BPXO_L），之前是聽診（BPX），兩者不同。
3. 減重藥條件再加睡眠呼吸中止替代定義（Wong 2025：打鼾或打呼／呼吸暫停每週 ≥ 3 晚；SLQ030／SLQ040 只有 2015–2018 有），只在 2015–2018。
4. 只用 2013–2018（2021–2023 的自報身高體重可在家或以電話問，之前只在家）。
5. 主要估計值的 Taylor 線性化 SE（層內有放回 PSU 近似）與 Rao–Wu bootstrap SE 比較。

沿用 analysis.py 前段（含 Rao–Wu bootstrap）。輸出 out/r7_sensitivity.csv、out/r7_variance.csv。
"""
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
src = (HERE / "analysis.py").read_text()
exec(compile(src[:src.index("t1 = [summary(r)")], "analysis.py", "exec"))
N = len(d); w = d.w.values; Wf = WB.astype(np.float64)
pctl = lambda bs: np.percentile(bs, [2.5, 97.5])
ALLM = np.ones(N, bool)


def rate(num, den, W=None):
    num = num & den
    if W is None:
        return 100 * (w @ num) / (w @ den)
    return 100 * (W @ num.astype(float)) / (W @ den.astype(float))


def add(rows, name, rule, m, s, mask=ALLM):
    v, b = rate(m != s, mask), rate(m != s, mask, Wf)
    rows.append({"analysis": name, "rule": rule, "n": int(mask.sum()), "discordant": v, "lo": pctl(b)[0], "hi": pctl(b)[1]})


# ---------------------------------------------------------------- 實測條件
def lab(cyc, stem, cols):
    f = (DAT / f"{stem}_L.xpt") if cyc == "2021-2023" else (DAT / f"{stem}_{WHQ[cyc]}.xpt")
    x = pd.read_sas(f)
    return x[["SEQN"] + [c for c in cols if c in x.columns]]


parts = []
for cyc in CYCLES:
    if cyc == "2021-2023":
        bp = lab(cyc, "BPXO", ["BPXOSY1", "BPXOSY2", "BPXOSY3", "BPXODI1", "BPXODI2", "BPXODI3"])
        sy, di = bp.filter(like="BPXOSY"), bp.filter(like="BPXODI")
    else:
        bp = lab(cyc, "BPX", ["BPXSY1", "BPXSY2", "BPXSY3", "BPXSY4", "BPXDI1", "BPXDI2", "BPXDI3", "BPXDI4"])
        sy, di = bp.filter(like="BPXSY"), bp.filter(like="BPXDI").where(lambda x: x > 0)   # 聽診舒張壓 0 視為缺值
    bp = pd.DataFrame({"SEQN": bp.SEQN, "sbp": sy.mean(1), "dbp": di.mean(1)})
    tc, gh = lab(cyc, "TCHOL", ["LBXTC"]), lab(cyc, "GHB", ["LBXGH"])
    x = bp.merge(tc, on="SEQN", how="outer").merge(gh, on="SEQN", how="outer")
    if cyc in ("2015-2016", "2017-2018"):
        sl = pd.read_sas(DAT / f"SLQ_{WHQ[cyc]}.xpt")[["SEQN", "SLQ030", "SLQ040"]]
        x = x.merge(sl, on="SEQN", how="left")
    parts.append(x)
L = pd.concat(parts, ignore_index=True)
d = d.merge(L, on="SEQN", how="left")
assert len(d) == N

hyp_m = d.BPQ020.eq(1) | (d.sbp >= 130) | (d.dbp >= 80)
chol_m = d.BPQ080.eq(1) | (d.LBXTC >= 200)
dm_m = d.DIQ010.eq(1) | (d.LBXGH >= 6.5)
cvd = d[["MCQ160B", "MCQ160C", "MCQ160D", "MCQ160E", "MCQ160F"]].eq(1).any(axis=1)
osa = d.SLQ030.isin([2, 3]) | d.SLQ040.isin([2, 3])          # 2 = 3–4 晚／週（occasionally），3 = 5 晚以上（frequently）
com_m = (hyp_m | chol_m | dm_m | cvd).values
met_m = (hyp_m | dm_m).values
bm, bs = d.bmi_meas.values, d.bmi_self.values

rows = []
aom = lambda b, c: (b >= 30) | ((b >= 27) & c)
mbs = lambda b, c: (b >= 35) | ((b >= 30) & c)
add(rows, "Primary: self-reported conditions", "AOM", d.AOM_meas.values, d.AOM_self.values)
add(rows, "Primary: self-reported conditions", "MBS", d.MBS_meas.values, d.MBS_self.values)
dm_only = d.DIQ010.eq(1).values
add(rows, "Metabolic disease for surgery: diabetes only", "MBS", mbs(bm, dm_only), mbs(bs, dm_only))
add(rows, "Conditions self-reported or measured (blood pressure, total cholesterol, HbA1c)", "AOM", aom(bm, com_m), aom(bs, com_m))
add(rows, "Conditions self-reported or measured (blood pressure, total cholesterol, HbA1c)", "MBS", mbs(bm, met_m), mbs(bs, met_m))
c1518 = d.cycle.isin(["2015-2016", "2017-2018"]).values
com_osa = d.com.values | osa.values
add(rows, "2015–2018: primary conditions", "AOM", d.AOM_meas.values, d.AOM_self.values, c1518)
add(rows, "2015–2018: conditions plus snoring or breathing pauses ≥ 3 nights a week", "AOM", aom(bm, com_osa), aom(bs, com_osa), c1518)
pre = (d.cycle != "2021-2023").values
for r in ("AOM", "MBS"):
    add(rows, "2013–2018 only (self-report collected in the home)", r, d[f"{r}_meas"].values, d[f"{r}_self"].values, pre)
sens = pd.DataFrame(rows)
sens.to_csv(OUT / "r7_sensitivity.csv", index=False)

# 條件判定改變的人有多少（說明用）
extra = {"share_with_condition_selfreport": 100 * (w @ d.com.values) / w.sum(), "share_with_condition_self_or_measured": 100 * (w @ com_m) / w.sum()}

# ---------------------------------------------------------------- Taylor 線性化 vs bootstrap
def taylor_se(y, den=ALLM):
    yy, dd = (y & den).astype(float), den.astype(float)
    R = (w @ yy) / (w @ dd)
    z = w * (yy - R * dd) / (w @ dd)
    t = pd.DataFrame({"st": d.stratum.values, "psu": d.psu.values, "z": z}).groupby(["st", "psu"]).z.sum().reset_index()
    v = 0.0
    for _, g in t.groupby("st"):
        nh = len(g)
        v += nh / (nh - 1) * ((g.z - g.z.mean()) ** 2).sum()
    return 100 * R, 100 * np.sqrt(v)


var = []
for r in ("AOM", "MBS"):
    y = d[f"{r}_meas"].values != d[f"{r}_self"].values
    est, se_t = taylor_se(y)
    bsv = rate(y, ALLM, Wf)
    var.append({"rule": r, "estimate": est, "se_taylor": se_t, "ci_taylor_lo": est - 1.96 * se_t, "ci_taylor_hi": est + 1.96 * se_t,
                "se_bootstrap": bsv.std(ddof=1), "ci_bootstrap_lo": pctl(bsv)[0], "ci_bootstrap_hi": pctl(bsv)[1]})
pd.DataFrame(var).to_csv(OUT / "r7_variance.csv", index=False)
pd.set_option("display.width", 220)
print(sens.round(2).to_string()); print(pd.DataFrame(var).round(3).to_string()); print(extra)
