"""第六輪（外部意見 2026-10-23）：
1. 針對性灌水補「規則公開、刻意報到實測區上緣之外」：灌水者的自報 BMI 設為門檻 + 1.1（−2 到 +1 區外、−2 到 +2 區內）與門檻 + 2.1（兩區皆外）。
2. 機轉：各族群 |自報 − 實測 BMI| > 2 kg/m² 的比例，以及不對稱規則（k = 1）下殘餘誤判者的 |誤差| 中位數與 > 2 的比例。

沿用 analysis.py 前段；灌水者的隨機抽取與 analysis5 相同（同一 seed）。輸出 out/r6_targeted_beyond.csv、out/r6_error_by_race.csv。
"""
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
src = (HERE / "analysis.py").read_text()
exec(compile(src[:src.index("t1 = [summary(r)")], "analysis.py", "exec"))
N = len(d); w = d.w.values
TH = {"AOM": np.where(d.com.values, 27.0, 30.0), "MBS": np.where(d.met.values, 30.0, 35.0)}
ALLM = np.ones(N, bool)
rate = lambda num, den: 100 * (w @ (num & den)) / (w @ den)
U = np.random.default_rng(20261024).random(N)

rows = []
for r, f in RULES.items():
    if r not in ("AOM", "MBS"):
        continue
    m = d[f"{r}_meas"].values; T = TH[r]; b0 = d.bmi_self.values
    for X_ in (1.0, 2.0, 3.0):
        for p in (0.25, 0.5, 1.0):
            gm = (b0 < T) & (b0 >= T - X_) & (U < p)
            for land, lab in [(0.1, "just above threshold (+0.1)"), (1.1, "beyond the −2 to +1 zone (+1.1)"), (2.1, "beyond the −2 to +2 zone (+2.1)")]:
                bs = np.where(gm, T + land, b0); s = f(bs, d); dl = bs - T
                row = {"rule": r, "within_kg_m2": X_, "share_inflating": p, "landing": lab, "inflators_pct": rate(gm, ALLM),
                       "discordant": rate(m != s, ALLM), "self_only": rate(~m & s, ALLM)}
                for nm, z in [("m2p1", (dl > -2) & (dl < 1)), ("m2p2", (dl > -2) & (dl < 2))]:
                    fin = np.where(z, m, s)
                    row[f"{nm}_measured"] = rate(z, ALLM); row[f"{nm}_residual"] = rate(m != fin, ALLM); row[f"{nm}_self_only"] = rate(~m & fin, ALLM)
                rows.append(row)
pd.DataFrame(rows).to_csv(OUT / "r6_targeted_beyond.csv", index=False)

RACE = [("Non-Hispanic White", [3]), ("Non-Hispanic Black", [4]), ("Hispanic", [1, 2]), ("Non-Hispanic Asian", [6]), ("Other or multiracial", [7]), ("All adults", [1, 2, 3, 4, 6, 7])]
err = np.abs(d.bmi_self.values - d.bmi_meas.values)
er = []
for r in ("AOM", "MBS"):
    m = d[f"{r}_meas"].values; s = d[f"{r}_self"].values; dl = d.bmi_self.values - TH[r]
    z = (dl > -2) & (dl < 1); resid = m != np.where(z, m, s)
    for lab, codes in RACE:
        g = np.isin(d.RIDRETH3.values, codes); gr = g & resid
        o = np.argsort(err[gr]); cw = np.cumsum(w[gr][o]) / w[gr].sum()
        er.append({"rule": r, "group": lab, "error_gt2_pct_all": rate(err > 2, g), "error_gt1_pct_all": rate(err > 1, g),
                   "residual_n": int(gr.sum()), "residual_median_abs_error": float(err[gr][o][np.searchsorted(cw, 0.5)]),
                   "residual_error_gt2_pct": rate(err > 2, gr)})
pd.DataFrame(er).to_csv(OUT / "r6_error_by_race.csv", index=False)
pd.set_option("display.width", 220)
t = pd.DataFrame(rows); print(t[t.share_inflating == 1.0].round(2).to_string()); print(pd.DataFrame(er).round(2).to_string())
