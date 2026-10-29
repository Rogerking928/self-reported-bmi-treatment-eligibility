"""圖表本（一份 Tables_and_Figures.docx）＋ 每張圖 700 dpi PNG ＋ Source_data.xlsx，全放同一個交付資料夾；不產 PDF。
版式照 Bcc 那篇：Letter 橫向、左右 1.0／上下 0.9 吋、每項一頁、標題 TNR 11.5 pt 粗體、表 8 pt 三線黑框、表註 9 pt、圖說 9.5 pt 正體。
排版函式在 book_base.py（沿用作者前一篇已交付過的版式）。

第三輪（外部意見 2026-10-23）：主文 2 表 3 圖（特徵、分類表現；機制、誰被分錯、實測策略的取捨前緣），補充 6 表 4 圖。
補充編號依主文圖表說明第一次提及的順序：Table 1 → 圖 S1、表 S1、表 S2；Figure 1 → 圖 S2；Figure 2 → 表 S3、表 S4、圖 S3；
Figure 3 → 表 S1、表 S5、圖 S4、表 S6–S10（第七輪：原表 S5 拆成模型 S5 與策略 S6，其後順延）。

  python3 code/book.py [交付資料夾]
"""
import glob, json, os, shutil, sys
from pathlib import Path
import numpy as np
import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Font as XFont, PatternFill, Alignment

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "out"
DEST = os.environ.get("DELIVERABLE_DIR", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "deliverable"))
import book_base  # noqa: E402

D = "–"
rd = lambda f, **k: pd.read_csv(OUT / f, **k)
def c3(v, lo, hi, nd=1):
    """估計值（CI）；區間含負值時用 to 分隔，避免「−1.2–0.9」被讀成負號；−0.0 寫成 0.0。"""
    f = lambda x: f"{x:.{nd}f}".replace("-0." + "0" * nd, "0." + "0" * nd).replace("-", "−")
    sep = " to " if lo < 0 else D
    return f"{f(v)} ({f(lo)}{sep}{f(hi)})"
UNIT = "Unit: participant."
DESIGN = ("Weighted estimates; 95% confidence intervals from 1,000 rescaling bootstrap resamples of primary sampling units within strata (Rao–Wu, n_h − 1 units drawn per "
          "stratum).")
RULES = ("Anti-obesity medication: BMI ≥ 30 kg/m², or ≥ 27 kg/m² with at least one self-reported weight-related condition (hypertension, high cholesterol, "
         "diabetes or cardiovascular disease). Metabolic and bariatric surgery: BMI ≥ 35 kg/m², or ≥ 30 kg/m² with self-reported diabetes or hypertension.")
MINUS = lambda v: str(v).replace("-", "−") if str(v).startswith("-") else v


def items():
    t1 = rd("r2_table1.csv"); t2 = rd("r2_table2.csv").set_index("rule"); x2 = rd("r3_table2_extra.csv").set_index("rule")
    sg = rd("r2_subgroups.csv", keep_default_na=False); se = rd("r2_sensitivity.csv"); q2 = json.load(open(OUT / "r2_summary.json"))
    q3 = json.load(open(OUT / "r3_summary.json")); kt = rd("r3_kitagawa.csv"); coef = rd("r3_model_coef.csv"); perf = rd("r3_model_perf.csv")
    ops = rd("r3_operating_points.csv"); t1m = rd("r3_table1_mbs.csv")
    A, M = t2.loc["AOM"], t2.loc["MBS"]; XA, XM = x2.loc["AOM"], x2.loc["MBS"]
    rows1 = [["Characteristic", "All adults", "Concordant eligibility", "Discordant eligibility"]] + [[MINUS(v) for v in row] for row in t1.values.tolist()]
    pct = lambda r, k: c3(r[k], r[k + "_lo"], r[k + "_hi"])
    P4 = rd("r4_population.csv").set_index(["rule", "scope", "quantity"])
    popc = lambda r, sc, q: c3(P4.loc[(r, sc, q), "millions"], P4.loc[(r, sc, q), "lo"], P4.loc[(r, sc, q), "hi"])
    tot = lambda sc: P4.loc[("AOM", sc, "total"), "millions"]
    rows2 = [["", "Anti-obesity medication", "Metabolic and bariatric surgery"],
             ["Cross-classification, n (unweighted)", "", ""],
             ["   Eligible by both", f"{int(A.n_mm):,}", f"{int(M.n_mm):,}"],
             ["   Eligible by measured values only", f"{int(A.n_m_only):,}", f"{int(M.n_m_only):,}"],
             ["   Eligible by self-reported values only", f"{int(A.n_s_only):,}", f"{int(M.n_s_only):,}"],
             ["   Eligible by neither", f"{int(A.n_neither):,}", f"{int(M.n_neither):,}"],
             ["Self-report against measurement, % (95% CI)", "", ""],
             ["   Sensitivity", pct(A, "sensitivity"), pct(M, "sensitivity")],
             ["   Specificity", pct(A, "specificity"), pct(M, "specificity")],
             ["   Positive predictive value", pct(A, "ppv"), pct(M, "ppv")],
             ["   Negative predictive value", pct(A, "npv"), pct(M, "npv")],
             ["   Discordant classification", pct(A, "discordant"), pct(M, "discordant")],
             ["      Eligible by measured values only", pct(A, "measured_only"), pct(M, "measured_only")],
             ["      Eligible by self-reported values only", pct(A, "self_only"), pct(M, "self_only")],
             ["   Survey-weighted Cohen's κ (95% CI)", c3(A.kappa, A.kappa_lo, A.kappa_hi, 2), c3(M.kappa, M.kappa_lo, M.kappa_hi, 2)],
             ["   Ratio, measured-only to self-only (95% CI)", c3(XA.ratio, XA.ratio_lo, XA.ratio_hi), c3(XM.ratio, XM.ratio_lo, XM.ratio_hi)],
             *[row for sc in ("all cycles (average)", "2021–2023 cycle") for row in
               [[f"US adults, {sc}, millions (95% CI)", "", ""]] +
               [[lab, popc("AOM", sc, q), popc("MBS", sc, q)] for lab, q in
                [("   Eligible by measured values", "eligible_measured"), ("   Discordant classification", "discordant"),
                 ("      Eligible by measured values only", "measured_only"), ("      Eligible by self-reported values only", "self_only")]]]]
    qa, qm = q2["quadrants"]["AOM"], q2["quadrants"]["MBS"]
    asm = q3["asian_mbs_specific"]
    _tv = rd("r7_variance.csv").set_index("rule")
    global tl
    tl = lambda r: c3(_tv.loc[r, "estimate"], _tv.loc[r, "ci_taylor_lo"], _tv.loc[r, "ci_taylor_hi"])
    _fr = rd("r4_frontier.csv"); cal = rd("r3_calibration.csv").set_index("rule")
    pb = {r: _fr[(_fr.rule == r) & _fr.strategy.str.startswith("Probability") & np.isclose(_fr.value, 0.5)].residual.iloc[0] for r in ("AOM", "MBS")}   # 機率式、0% 實測
    p0 = {r: _fr[(_fr.rule == r) & (_fr.strategy == "Asymmetric 2:1") & np.isclose(_fr.value, 0)].residual.iloc[0] for r in ("AOM", "MBS")}       # 只用自報
    main = [
        {"kind": "table", "caption": "Table 1. Characteristics of participants, overall and by concordance of eligibility for anti-obesity medication",
         "rows": rows1,
         "note": f"{UNIT} US adults aged 18 years or older, NHANES 2013–2018 and 2021–2023 (selection in Supplementary Figure S1). Weighted percentages or means "
                 "(unweighted n or SD); weights are mobile examination centre weights divided by 4. Concordant: eligible, or not eligible, by both self-reported and measured "
                 "height and weight; discordant: eligible by one and not the other. " + RULES + " Definitions in Supplementary Table S1; the same table for "
                 "metabolic and bariatric surgery in Supplementary Table S2."},
        {"kind": "table", "caption": "Table 2. Eligibility from self-reported against measured height and weight",
         "rows": rows2,
         "note": f"{UNIT} n = {int(A.n_mm + A.n_m_only + A.n_s_only + A.n_neither):,}. Measured height and weight are the reference. {DESIGN} "
                 f"Population counts are weighted sums over the analysed participants: over all cycles, an average over the survey years (weight total {tot('all cycles (average)'):.1f} million); "
                 f"for 2021–2023, that cycle's weights alone (weight total {tot('2021–2023 cycle'):.1f} million). Because participants were excluded (Supplementary Figure S1), the weight totals "
                 "are below the US adult population and the counts are lower bounds. For counts, each bootstrap replicate's weights were "
                 "re-scaled within cycle to the full-sample weight total (re-raking), so that their uncertainty matches that of the corresponding percentage. Cohen's κ is computed with survey weights (unweighted-agreement κ, "
                 "not an ordinal weighted κ). Ratio: weighted adults eligible by measured values only divided by those eligible by self-reported values only."},
        {"kind": "figure", "path": OUT / "Figure_1.png", "caption": "Figure 1. How self-reported height and weight shift eligibility",
         "note": "Top row, anti-obesity medication; bottom row, metabolic and bariatric surgery. "
                 "Blue, anti-obesity medication; vermilion, metabolic and bariatric surgery. "
                 "(A, D) Self-reported against measured BMI, weighted hexagonal bins (log colour scale, thousands of adults); values below 18 or above 48 kg/m² are pooled in "
                 "the edge bins (the source data tabulate the same values in 1 kg/m² squares); dotted line, identity; black lines at 30 (A) and 35 kg/m² (D). Bold figures: weighted percentage of all adults in each discordant quadrant judged at that single threshold "
                 f"({qa['meas_ge_self_lt']:.1f}% and {qa['meas_lt_self_ge']:.1f}% in A; {qm['meas_ge_self_lt']:.1f}% and {qm['meas_lt_self_ge']:.1f}% in D); "
                 "they differ from Table 2, which applies the full rule including qualifying conditions. "
                 "(B, E) Weighted percentage classified eligible by self-report, by measured BMI in 0.5 kg/m² bins (bins with fewer than 10 participants not shown), "
                 "with 95% bootstrap bands; solid, with a qualifying condition (any weight-related condition in B; diabetes or hypertension in E); dashed and paler, without; "
                 "dotted steps, the classification that measured values would give. "
                 "(C, F) Within each 0.5 kg/m² bin of measured BMI, weighted percentage eligible by measured values only (upward, darker) and by self-reported values only "
                 "(downward, paler), shown for measured BMI 20 to 40 kg/m² (a few bins lie above 40); "
                 "the grey strip shows the share of adults in each bin (darker, more adults). Distributions of the reporting differences in Supplementary Figure S2. " + UNIT},
        {"kind": "figure", "path": OUT / "Figure_2.png", "caption": "Figure 2. Discordant eligibility by subgroup, crude and standardised to the measured-BMI distribution",
         "note": "Blue, anti-obesity medication; vermilion, metabolic and bariatric surgery. Filled circles, crude weighted percentage; open squares, percentage standardised to the distribution of measured BMI (1 kg/m² bands) and qualifying-condition "
                 "status of all adults, which removes differences due to how many adults in a subgroup lie near a threshold. For anti-obesity medication, the "
                 "weight-related-condition rows are shown crude only, because the standardisation strata contain condition status and the standardised value equals the crude "
                 f"value by construction. Triangle: non-Hispanic Asian adults with the Asian-specific surgery threshold (BMI ≥ 27.5 kg/m²; ASMBS and IFSO 2022), "
                 f"{asm['discordant']:.1f}% ({asm['lo']:.1f}{D}{asm['hi']:.1f}). {DESIGN} Vertical lines, all adults. Text columns: crude / standardised. When a subgroup does not occupy every stratum (adults without a qualifying condition occupy only the no-condition strata), the standardised value uses the strata present, reweighted; the share of the all-adult population in those strata is listed in Supplementary Table S3. "
                 "Full values and direction of discordance in Supplementary Table S3; decomposition of subgroup differences into BMI distribution and within-band "
                 "components in Supplementary Table S4; by survey cycle in Supplementary Figure S3. " + UNIT},
        {"kind": "figure", "path": OUT / "Figure_3.png", "caption": "Figure 3. Measuring height and weight near the thresholds: trade-off and equity",
         "note": "(A, B) Validation cycle 2021–2023: share of adults measured against residual discordant eligibility when height and weight are measured for adults selected by "
                 "each strategy and the others are classified from self-report. Asymmetric 2:1 (thick line): measure when self-reported BMI is between 2k below and k above the "
                 "person's applicable threshold; drawn in the rule colour (blue, medication; vermilion, surgery), other strategies in grey; diamond, k = 1 kg/m²; band, 95% bootstrap "
                 "interval. Strategy definitions in Supplementary Table S1. The probability-based curve starts lower than the others at 0% measured "
                 f"({pb['AOM']:.2f}% vs {p0['AOM']:.2f}% for medication; {pb['MBS']:.2f}% vs {p0['MBS']:.2f}% for surgery) because adults who are not measured are classified "
                 "by the predicted probability from the correction model (mean and log-variance; Supplementary Table S5) rather than by self-report; the mean-only correction "
                 f"in Supplementary Figure S4 gives {cal.loc['AOM', 'corrected']:.2f}% for medication. Operating points and paired comparisons in Supplementary Table S6. "
                 "(C) All four cycles (the rule has no fitted parameter): open symbols, discordant eligibility with self-report only; filled symbols, residual discordance "
                 "under the asymmetric rule with k = 1; bars, 95% CI; circles, medication; squares, surgery; White, Black and Asian denote non-Hispanic adults. "
                 "Group values in Supplementary Table S7; absolute and relative gaps and race-free targeting in Supplementary Table S8; over-reporting scenarios in "
                 "Supplementary Table S9; sensitivity analyses in Supplementary Table S10. " + UNIT},
    ]
    defs = [["Item", "Definition", "Source variable"],
            ["Population", "Adults ≥ 18 years examined in the mobile examination centre; pregnant participants and measurements with an examiner comment excluded", "RIDAGEYR, RIDSTATR, RIDEXPRG, BMIWT, BMIHT"],
            ["Self-reported height and weight", "Interview before the examination, in the home (2013–2018) or in the home or by telephone (2021–2023); inches and pounds; plausible ranges 48–90 in and 75–600 lb", "WHD010, WHD020"],
            ["Measured height and weight", "Mobile examination centre; centimetres and kilograms", "BMXHT, BMXWT"],
            ["Weight-related condition", "Self-reported physician-diagnosed hypertension, high cholesterol, diabetes, or congestive heart failure, coronary heart disease, angina, heart attack or stroke", "BPQ020, BPQ080, DIQ010, MCQ160B–F"],
            ["Anti-obesity medication eligibility", "BMI ≥ 30 kg/m², or ≥ 27 kg/m² with ≥ 1 weight-related condition (product labels for semaglutide (Wegovy) and tirzepatide (Zepbound): adults with obesity, or with overweight and at least one weight-related comorbid condition; BMI cut-offs as in the registration trials described in the labels)", "—"],
            ["Metabolic and bariatric surgery eligibility", "BMI ≥ 35 kg/m², or ≥ 30 kg/m² with diabetes or hypertension (ASMBS and IFSO 2022, which do not define metabolic disease; counting hypertension is the author's interpretation, and diabetes alone is a sensitivity analysis)", "—"],
            ["Asian-specific surgery threshold", "Non-Hispanic Asian adults: BMI ≥ 27.5 kg/m² (ASMBS and IFSO 2022)", "RIDRETH3"],
            ["Asian-specific medication threshold (exploratory)", "Non-Hispanic Asian adults: BMI ≥ 27.5, or ≥ 24.5 kg/m² with a weight-related condition, i.e. the label thresholds lowered by 2.5 kg/m² as NICE TA875 advises for people of Asian and some other family backgrounds; not a US label criterion", "RIDRETH3"],
            ["Discordant classification", "Eligible by one source and not by the other; split into eligible by measured values only and by self-reported values only", "—"],
            ["Standardisation", "Direct standardisation of subgroup discordance to the all-adult distribution of measured BMI (1 kg/m² bands, 15–50) × qualifying-condition status; bands absent in a subgroup dropped and weights renormalised", "—"],
            ["Decomposition", "Kitagawa: crude difference = Σ (p₁ − p₂)(r₁ + r₂)/2 [composition] + Σ (r₁ − r₂)(p₁ + p₂)/2 [within-band rates], over the same strata", "—"],
            ["Correction model", "Weighted linear regression of measured BMI on self-reported BMI, age, sex, race and ethnicity and self-reported weight ending in 0 or 5; log squared residual modelled on the same terms; fitted to 2013–2018, applied to 2021–2023", "—"],
            ["Measurement strategies", "Measure when self-reported BMI is within ± k (symmetric) or between −r·k and + k (asymmetric; r = 2 primary, 1.5 and 3 sensitivity) of the person's applicable threshold (27 or 30 kg/m² for medication; 30 or 35 kg/m² for surgery); probability-based: measure when the predicted probability that measured BMI is at or above the threshold lies in [p, 1 − p], from the correction model with its log-variance model and the empirical distribution of standardised training residuals (residuals are heavy-tailed)", "—"],
            ["Race-free targeting", "As probability-based, with models excluding race and ethnicity (self-reported BMI, age, sex, weight ending in 0 or 5), cross-fitted between 2013–2018 and 2021–2023 and matched to the asymmetric rule on share measured", "—"],
            ["Survey design", "Mobile examination centre weights divided by 4 (four 2-year cycles); 95% CIs from 1,000 rescaling bootstrap resamples (Rao–Wu): in each stratum n_h − 1 primary sampling units drawn with replacement and their weights multiplied by n_h/(n_h − 1) (all strata have 2 units); percentile intervals", "WTMEC2YR, SDMVSTRA, SDMVPSU"],
            ["Not available", "Obstructive sleep apnoea and other weight-related conditions (a snoring proxy exists only in 2015–2018); time between interview and examination; mode of the 2021–2023 interview (in the home or by telephone) for each participant", "—"]]
    t1mr = [["Characteristic", "All adults", "Concordant eligibility", "Discordant eligibility"]] + t1m.values.tolist()
    RL = {"AOM": "Medication", "MBS": "Surgery"}
    sgr = [["Group", "Subgroup", "n", "Rule", "Crude, % (95% CI)", "Standardised, % (95% CI)", "Measured only, %", "Self-report only, %", "Within 1.5 kg/m² of threshold, %", "Reference strata covered, %"]]
    for r in ("AOM", "MBS"):
        for x in sg[sg.rule == r].itertuples():
            stdv = "= crude by construction" if (r == "AOM" and x.group == "Weight-related condition") else c3(x.standardized, x.std_lo, x.std_hi)
            sgr.append([x.group, x.subgroup.replace("-", "–") if x.group == "Survey cycle" else x.subgroup, f"{x.n:,}", RL[r], c3(x.crude, x.crude_lo, x.crude_hi),
                        stdv, f"{x.measured_only:.1f}", f"{x.self_only:.1f}", f"{x.near_threshold_pct:.1f}", f"{x.strata_used_pct_of_population:.1f}"])
    ktr = [["Comparison", "Rule", "Crude difference, pp (95% CI)", "BMI-distribution component, pp (95% CI)", "Within-band component, pp (95% CI)", "Share due to BMI distribution, %"]]
    for x in kt.itertuples():
        # 判斷用表上顯示的（兩位小數）粗差值，避免表上寫 −0.98 卻給出比例
        ktr.append([x.comparison, RL[x.rule], c3(x.crude_diff, x.crude_lo, x.crude_hi, 2), c3(x.composition, x.comp_lo, x.comp_hi, 2),
                    c3(x.within_stratum, x.rate_lo, x.rate_hi, 2),
                    "—" if abs(round(x.crude_diff, 2)) < 1 else ("opposite directions" if x.composition * x.within_stratum < 0 else f"{x.composition_share_pct:.0f}")])
    TERM = {"const": "Intercept", "self_bmi": "Self-reported BMI, per kg/m²", "age10": "Age, per 10 years (centred at 50)", "women": "Women",
            "black": "Non-Hispanic Black", "hispanic": "Hispanic", "asian": "Non-Hispanic Asian", "other_race": "Other or multiracial",
            "heaped_weight": "Self-reported weight ending in 0 or 5"}
    mrows = [["Term", "Mean model, kg/m² (95% CI)", "Log-variance model (95% CI)"]]
    cm, cv = coef[coef.model == "mean"].set_index("term"), coef[coef.model == "log variance"].set_index("term")
    for t in cm.index:
        mrows.append([TERM[t], c3(cm.loc[t, "coef"], cm.loc[t, "lo"], cm.loc[t, "hi"], 3), c3(cv.loc[t, "coef"], cv.loc[t, "lo"], cv.loc[t, "hi"], 3)])
    mrows += [["Performance", "", ""]]
    for p in perf.itertuples():
        mrows += [[f"   {p.set} (n = {p.n:,}): mean error, self-reported → corrected", f"{p.mean_error_self:+.2f} → {p.mean_error_corrected:+.2f}".replace("-", "−"), ""],
                  [f"   {p.set}: mean absolute error, self-reported → corrected", f"{p.mae_self:.2f} → {p.mae_corrected:.2f}", ""],
                  [f"   {p.set}: residual SD", f"{p.sd_residual:.2f}", ""]]
    cov = q3["model"]["coverage_validation"]; cov4 = json.load(open(OUT / "r4_summary.json"))["coverage_validation_empirical"]
    mrows += [["   Validation: coverage of 50%, 80% and 95% prediction intervals, normal residuals", f"{cov['pi50'] * 100:.1f}%, {cov['pi80'] * 100:.1f}%, {cov['pi95'] * 100:.1f}%", ""],
              ["   Validation: coverage, empirical residual quantiles (used in Figure 3)", f"{cov4['pi50_empirical'] * 100:.1f}%, {cov4['pi80_empirical'] * 100:.1f}%, {cov4['pi95_empirical'] * 100:.1f}%", ""],
              ["   R², training", f"{q3['model']['r2']:.3f}", ""]]
    orows = [["Strategy", "Value", "Parameter or difference"],
             ["Measurement needed to reach a residual discordance, validation 2021–2023", "% measured for ≤ 3%, ≤ 2%, ≤ 1%", "Parameter (k or p)"]]
    ops4 = rd("r4_operating_points.csv"); pr4 = rd("r4_paired.csv")
    for r in ("AOM", "MBS"):
        for st in ("Asymmetric 2:1", "Asymmetric 1.5:1", "Asymmetric 3:1", "Symmetric ± k", "Probability-based (empirical)"):
            g = ops4[(ops4.rule == r) & (ops4.strategy == st)].set_index("target")
            orows.append([f"   {RL[r]}, {st}", ", ".join(f"{g.loc[tg, 'measured_pct']:.1f}" for tg in (3, 2, 1)), ", ".join(f"{g.loc[tg, 'value']:g}" for tg in (3, 2, 1))])
    orows += [["Residual discordance at the share measured by the 2:1 rule with k = 1, % (strategy vs 2:1 rule); difference, pp (95% CI)", "", ""]]
    for x in pr4.itertuples():
        orows.append([f"   {RL[x.rule]} ({x.measured_share:.1f}% measured), {x.strategy}", f"{x.residual_at_share:.2f} vs {x.residual_asym2:.2f}",
                      c3(x.difference_vs_asym2, x.diff_lo, x.diff_hi, 2)])
    eqd = rd("r4_equity.csv"); e6 = rd("r6_error_by_race.csv").set_index(["rule", "group"])
    eqr = [["Group", "Rule", "n", "Self-report only, % (95% CI)", "Asymmetric rule k = 1: measured, %", "Asymmetric rule k = 1: residual, % (95% CI)", "k for ≤ 2% residual",
            "Measured at that k, %", "|Self-reported − measured BMI| > 2 kg/m², % of group", "Residual misclassified: median |error|, kg/m²"]]
    for x in eqd.itertuples():
        e_ = e6.loc[(x.rule, x.group)]
        eqr.append([x.group, RL[x.rule], f"{x.n:,}", c3(x.residual_none, x.res_none_lo, x.res_none_hi), f"{x.measured_k1:.1f}", c3(x.residual_k1, x.res_k1_lo, x.res_k1_hi),
                    f"{x.k_for_2pct:g}", f"{x.measured_for_2pct:.1f}", f"{e_.error_gt2_pct_all:.1f}", f"{e_.residual_median_abs_error:.1f}"])
    gp = rd("r5_gaps.csv"); e5 = rd("r5_equity.csv")
    STN = ["Self-report only", "Asymmetric rule, k = 1", "Race-free probability, same share measured", "Wider zone for weights ending in 0 or 5, same share measured"]
    gpr = [["Rule", "Strategy", "Share measured, %", "Residual, all adults, %", "Group vs non-Hispanic White", "Residual in group, %", "Absolute gap, pp (95% CI)", "Ratio (95% CI)"]]
    for r in ("AOM", "MBS"):
        for st in STN:
            a_ = e5[(e5.rule == r) & (e5.strategy == st) & (e5.group == "All adults")].iloc[0]
            for x in gp[(gp.rule == r) & (gp.strategy == st)].itertuples():
                gr = e5[(e5.rule == r) & (e5.strategy == st) & (e5.group == x.group)].iloc[0]
                gpr.append([RL[r], st, f"{a_.measured:.1f}", f"{a_.residual:.2f}", x.group, f"{gr.residual:.2f}", c3(x.abs_gap_pp, x.abs_lo, x.abs_hi, 2),
                            c3(x.ratio, x.ratio_lo, x.ratio_hi, 2)])
    t6 = rd("r6_targeted_beyond.csv"); t6 = t6[t6.share_inflating.isin([0.5, 1.0])]
    u5 = rd("r5_targeted.csv"); u5 = u5[u5.within_kg_m2.isna()].set_index("rule")
    wid = rd("r4_whatif.csv")
    wir = [["Scenario", "Rule", "Who over-reports", "Reported BMI after over-reporting", "Discordant, %", "Eligible by self-report only, %",
            "−2 to +1 zone: measured, %", "residual, %", "self-report only, %", "−2 to +2 zone: measured, %", "residual, %", "self-report only, %"],
           ["A. Uniform over-reporting", "", "", "", "", "", "", "", "", "", "", ""]]
    for x in wid.itertuples():
        wir.append(["", RL[x.rule], "Everyone", f"+{x.added_kg} kg", f"{x.discordant:.1f}", f"{x.self_only:.1f}", f"{x.asym_k1_measured:.1f}", f"{x.asym_k1_residual:.2f}",
                    f"{x.asym_k1_self_only:.2f}", f"{x.sym_up2_measured:.1f}", f"{x.sym_up2_residual:.2f}", f"{x.sym_up2_self_only:.2f}"])
    wir.append(["B. Targeted over-reporting", "", "", "", "", "", "", "", "", "", "", ""])
    LAND = {"just above threshold (+0.1)": "Threshold + 0.1", "beyond the −2 to +1 zone (+1.1)": "Threshold + 1.1", "beyond the −2 to +2 zone (+2.1)": "Threshold + 2.1"}
    for x in t6.itertuples():
        wir.append(["", RL[x.rule], f"{x.share_inflating:.0%} of those within {x.within_kg_m2:g} below", LAND[x.landing], f"{x.discordant:.1f}", f"{x.self_only:.1f}",
                    f"{x.m2p1_measured:.1f}", f"{x.m2p1_residual:.2f}", f"{x.m2p1_self_only:.2f}", f"{x.m2p2_measured:.1f}", f"{x.m2p2_residual:.2f}", f"{x.m2p2_self_only:.2f}"])
    RLS = {"AOM": "Anti-obesity medication", "MBS": "Metabolic and bariatric surgery", "BMI >= 30 only": "BMI ≥ 30 kg/m² alone"}
    srows = [["Analysis", "Eligibility rule", "n", "Discordant, % (95% CI)"]]
    for x in se.itertuples():
        if x.analysis.startswith("Asian-specific") and x.rule == "MBS":
            continue   # 手術的亞裔切點改用 ASMBS／IFSO 2022 的 ≥ 27.5（下一列）
        nm = x.analysis.replace("Asian-specific thresholds", "Asian-specific threshold (exploratory; label thresholds − 2.5 kg/m², NICE TA875 approach)") if x.rule == "AOM" else x.analysis
        srows.append([nm, RLS[x.rule], f"{x.n:,}", c3(x.discordant, x.lo, x.hi)])
    for x in rd("r7_sensitivity.csv").itertuples():
        if not x.analysis.startswith("Primary"):
            srows.append([x.analysis, RLS[x.rule], f"{x.n:,}", c3(x.discordant, x.lo, x.hi)])
    srows.append(["Asian-specific threshold (BMI ≥ 27.5 kg/m², ASMBS and IFSO 2022), non-Hispanic Asian adults", RLS["MBS"], f"{asm['n']:,}", c3(asm["discordant"], asm["lo"], asm["hi"])])
    # 亞裔各列排在一起（亞裔切點 → 標準切點比較）
    asian_rows = sorted([r_ for r_ in srows[1:] if "Asian" in r_[0]], key=lambda r_: (not r_[0].startswith("Asian-specific"), r_[1]))
    srows = [srows[0]] + [r_ for r_ in srows[1:] if "Asian" not in r_[0]] + asian_rows
    supp = [
        {"kind": "table", "caption": "Supplementary Table S1. Definitions", "rows": defs, "note": "NHANES, National Health and Nutrition Examination Survey."},
        {"kind": "table", "caption": "Supplementary Table S2. Characteristics of participants by concordance of eligibility for metabolic and bariatric surgery", "rows": t1mr,
         "note": f"{UNIT} As Table 1, with concordance defined by the surgery rule."},
        {"kind": "table", "caption": "Supplementary Table S3. Discordant eligibility by subgroup: crude, standardised and by direction", "rows": sgr,
         "note": f"{UNIT} {DESIGN} Standardisation as in Supplementary Table S1. Measured only and self-report only: weighted percentages of the subgroup. "
                 "Within 1.5 kg/m²: weighted percentage of the subgroup whose measured BMI lies within 1.5 kg/m² of their applicable threshold. "
                 "Measured only and self-report only may not sum to the crude value because of rounding. The 2019–2020 cycle was not completed and is not included."},
        {"kind": "table", "caption": "Supplementary Table S4. Decomposition of subgroup differences in discordant eligibility", "rows": ktr,
         "note": f"{UNIT} pp, percentage points. Kitagawa decomposition over measured-BMI bands (1 kg/m²) × qualifying-condition status (Supplementary Table S1); "
                 f"the two components sum to the crude difference. {DESIGN} Share not shown when the crude difference is below 1 percentage point; when the two components "
                 "have opposite signs the share is not a proportion and is shown as \u201copposite directions\u201d."},
        {"kind": "table", "caption": "Supplementary Table S5. Model correcting self-reported BMI", "rows": mrows,
         "note": f"{UNIT} Weighted least squares fitted to 2013–2018 (n = {int(perf.n.iloc[0]):,}); 95% CIs model-based. Error: predicted or self-reported minus measured BMI, kg/m². "
                 "Coverage: weighted share of validation-cycle adults whose measured BMI fell within the model's prediction interval; the residuals are heavy-tailed, so the "
                 "probability-based strategy uses the empirical distribution of standardised training residuals."},
        {"kind": "table", "caption": "Supplementary Table S6. Measurement needed under each strategy and paired comparisons", "rows": orows,
         "note": f"{UNIT} Validation cycle 2021–2023. Operating points: smallest measured share on the Figure 3 grid at which residual discordance reached each target. "
                 f"Paired comparisons: residual interpolated at the same measured share in each bootstrap replicate; negative differences favour the named strategy. {DESIGN}"},
        {"kind": "table", "caption": "Supplementary Table S7. Residual discordance by race and ethnicity under the asymmetric rule, all cycles", "rows": eqr,
         "note": f"{UNIT} All cycles. Asymmetric rule: measure when self-reported BMI is between 2k below and k above the person's applicable threshold. {DESIGN} "
                 "k for ≤ 2% residual: smallest k on a 0.125 kg/m² grid at which the group's residual discordance was 2% or less. Mechanism: "
                 f"{e6.loc[('AOM', 'All adults'), 'residual_error_gt2_pct']:.0f}% (medication) and {e6.loc[('MBS', 'All adults'), 'residual_error_gt2_pct']:.0f}% (surgery) of adults still "
                 "misclassified under the rule had a reporting error above 2 kg/m², i.e. their self-report fell outside the measurement zone; such errors were more common in "
                 "non-Hispanic Black and Hispanic adults, which no zone centred on the threshold can offset."},
        {"kind": "table", "caption": "Supplementary Table S8. Absolute and relative gaps in residual discordance, and race-free targeting of measurement, all cycles", "rows": gpr,
         "note": f"{UNIT} All cycles. Gaps compare each group with non-Hispanic White adults (difference in percentage points; ratio). The absolute gap is the primary measure: "
                 "when the reference residual is small (about 1% under the rule), ratios are inflated by the small denominator. Race-free probability: probability that "
                 "measured BMI is at or above the applicable threshold from models without race or ethnicity (self-reported BMI, age, sex, weight ending in 0 or 5; mean and "
                 "log-variance; empirical residual quantiles), cross-fitted between 2013–2018 and 2021–2023, with p set so that the share measured equals that of the asymmetric "
                 f"rule. Wider zone: −3 to +1.5 kg/m² for weights ending in 0 or 5 and a narrower symmetric-ratio zone for others, matched on share measured. {DESIGN}"},
        {"kind": "table", "caption": "Supplementary Table S9. Over-reporting scenarios", "rows": wir,
         "note": f"{UNIT} All cycles; NHANES respondents had no incentive to over-report, so these are simulations. (A) A fixed amount added to every self-reported weight. With "
                 f"2 kg added, the two directions of error nearly cancel at the population level (eligible for medication {u5.loc['AOM', 'uniform_2kg_eligible_self']:.1f}% by "
                 f"self-report vs {u5.loc['AOM', 'uniform_2kg_eligible_measured']:.1f}% measured) while {u5.loc['AOM', 'discordant']:.1f}% of adults are still misclassified. "
                 "(B) Among adults whose self-reported BMI lay within the stated distance (kg/m²) below their applicable threshold, the stated share (selected at random) report a "
                 "BMI just above the threshold (+0.1), or, if the measurement zone is known, just beyond its upper edge (+1.1 or +2.1). Zones are applied to the reported BMI. "
                 "Reports just above the threshold fall inside both zones and are measured; a known upper edge can be evaded by reporting just beyond it. Scenarios with 25% of adults "
                 "over-reporting are in the source data only."},
        {"kind": "table", "caption": "Supplementary Table S10. Sensitivity analyses", "rows": srows,
         "note": f"{UNIT} {DESIGN} Rounding alone: measured height and weight rounded to whole inches and pounds, the units of self-report, compared with unrounded measured values. "
                 "Asian-specific thresholds applied to both self-reported and measured values; the medication threshold applies the NICE TA875 reduction of 2.5 kg/m² to the US label "
                 "thresholds and is exploratory. Measured conditions: mean blood pressure ≥ 130/80 mmHg (auscultatory in 2013–2018, oscillometric in 2021–2023), total cholesterol "
                 "≥ 200 mg/dL or HbA1c ≥ 6.5%, each in addition to self-report, adapted from Wong et al. (who also counted antihypertensive and lipid-lowering medication and triglycerides ≥ 200 mg/dL, not used here); snoring, or snorting or stopping breathing, on at least 3 nights a week was asked "
                 "only in 2015–2018. In 2021–2023 the self-reported height and weight could be collected in the home or by telephone; earlier, in the home only. "
                 f"Variance check: with Taylor linearisation instead of the bootstrap, the primary estimates were {tl('AOM')}% for medication and {tl('MBS')}% for surgery."},
        {"kind": "figure", "path": OUT / "Figure_S1.png", "caption": "Supplementary Figure S1. Selection of participants",
         "note": "Exclusions applied in the order shown; each box gives the number remaining. NHANES 2013–2014, 2015–2016, 2017–2018 and 2021–2023. " + UNIT},
        {"kind": "figure", "path": OUT / "Figure_S2.png", "caption": "Supplementary Figure S2. Differences between self-reported and measured height and weight",
         "note": "(A, B) Weighted distribution of self-reported minus measured weight and height, by sex; values beyond the axis limits are pooled in the end bins (≤ and ≥ labels). "
                 "(C) Weighted distribution of the last digit of self-reported weight (pounds) and height (inches); dashed line, 10% expected without digit preference. " + UNIT},
        {"kind": "figure", "path": OUT / "Figure_S3.png", "caption": "Supplementary Figure S3. Discordant eligibility by survey cycle",
         "note": f"{DESIGN} Cycles placed at the midpoint of data collection (2021–2023 at 2022.6); the 2019–2020 cycle was suspended by the COVID-19 pandemic. Legend: linear trend in percentage points per 10 years "
                 "(weighted linear probability model; 95% bootstrap CI). Blue filled circles, anti-obesity medication; vermilion open squares, metabolic and bariatric surgery. " + UNIT},
        {"kind": "figure", "path": OUT / "Figure_S4.png", "caption": "Supplementary Figure S4. Statistical correction of self-reported BMI without measurement",
         "note": "Validation cycle 2021–2023: discordant eligibility from self-reported BMI and from self-reported BMI corrected by the mean model in Supplementary Table S5. "
                 "Difference: corrected minus self-reported, with 95% CI from paired bootstrap resamples. Correction removed the average under-reporting but not the "
                 "person-level error that drives misclassification near thresholds. " + UNIT},
    ]
    return main, supp


def model_extra():
    """表 S5 的覆蓋率與 R²（不在 r3_model_*.csv 裡）。"""
    q = json.load(open(OUT / "r3_summary.json"))["model"]; c4 = json.load(open(OUT / "r4_summary.json"))["coverage_validation_empirical"]
    rows = [{"model": "mean", "quantity": "R2_training", "value": q["r2"]}]
    rows += [{"model": "normal residuals", "quantity": f"coverage_validation_{k}", "value": v} for k, v in q["coverage_validation"].items()]
    rows += [{"model": "empirical residual quantiles", "quantity": f"coverage_validation_{k}", "value": v} for k, v in c4.items()]
    return pd.DataFrame(rows)


def binned_scatter():
    """圖 1A／D：加權分格人數（1 kg/m² × 1 kg/m²，18–48），不放個人層級資料。"""
    sc = rd("r2_scatter.csv"); e = np.arange(18, 49, 1.0)
    h, _, _ = np.histogram2d(sc.bmi_meas.clip(18, 47.99), sc.bmi_self.clip(18, 47.99), bins=[e, e], weights=sc.w / 1e3)
    n, _, _ = np.histogram2d(sc.bmi_meas.clip(18, 47.99), sc.bmi_self.clip(18, 47.99), bins=[e, e])
    ii, jj = np.nonzero(n)
    return pd.DataFrame({"measured_bmi_from": e[ii], "self_reported_bmi_from": e[jj], "participants": n[ii, jj].astype(int), "weighted_adults_thousands": h[ii, jj].round(1)})


def binned_diffs():
    """補充圖 S2A／B：依性別的加權分布（與圖同一組 bin；兩端合併）。"""
    df = rd("differences.csv"); out = []
    for col, (lo, hi) in [("weight_diff_kg", (-15, 15)), ("height_diff_cm", (-10, 10))]:
        bins = np.linspace(lo, hi, 61)
        for sx in ("Women", "Men"):
            x = df[df.sex == sx]; h, b = np.histogram(x[col].clip(lo, hi), bins=bins, weights=x.w)
            out += [{"panel": "A" if col.startswith("weight") else "B", "variable": col, "sex": sx, "bin_from": b[i], "bin_to": b[i + 1], "weighted_pct": 100 * h[i] / h.sum()}
                    for i in range(len(h))]
    return pd.DataFrame(out)


def source_data(path, main, supp):
    XF, XB, T = XFont(name="Arial", size=10), XFont(name="Arial", size=10, bold=True), XFont(name="Arial", size=11, bold=True)
    wb = Workbook(); wb.remove(wb.active); toc = wb.create_sheet("Contents"); toc["A1"] = "Source data (aggregate)"; toc["A1"].font = T
    cap = {it["caption"].split(". ")[0]: it["caption"] for it in main + supp}
    defs = supp[0]["rows"]
    sheets = [("Table 1", "Table 1", rd("r2_table1.csv")), ("Table 2", "Table 2", pd.concat([rd("r2_table2.csv"), rd("r3_table2_extra.csv")], axis=1)), ("Table 2 counts", "Table 2", rd("r4_population.csv")),
              ("Figure 1A,D quadrants", "Figure 1", pd.json_normalize(json.load(open(OUT / "r2_summary.json"))["quadrants"]).T.reset_index()),
              ("Figure 1A,D binned", "Figure 1", binned_scatter()), ("Figure 1B-F", "Figure 1", rd("r2_curves.csv")),
              ("Figure 2", "Figure 2", rd("r2_subgroups.csv", keep_default_na=False)),
              ("Figure 2 Asian threshold", "Figure 2", pd.DataFrame([json.load(open(OUT / "r3_summary.json"))["asian_mbs_specific"]])),
              ("Figure 3A-B", "Figure 3", rd("r4_frontier.csv")), ("Figure 3C", "Figure 3", rd("r4_equity.csv")),
              ("Table S1", "Supplementary Table S1", pd.DataFrame(defs[1:], columns=defs[0])),
              ("Table S2", "Supplementary Table S2", rd("r3_table1_mbs.csv")),
              ("Table S3", "Supplementary Table S3", rd("r2_subgroups.csv", keep_default_na=False)),
              ("Table S4", "Supplementary Table S4", rd("r3_kitagawa.csv")),
              ("Table S5", "Supplementary Table S5", pd.concat([rd("r3_model_coef.csv"), rd("r3_model_perf.csv"), model_extra()])),
              ("Table S6", "Supplementary Table S6", pd.concat([rd("r4_operating_points.csv"), rd("r4_paired.csv")])),
              ("Table S7", "Supplementary Table S7", rd("r4_equity.csv").merge(rd("r6_error_by_race.csv"), on=["rule", "group"])), ("Table S8", "Supplementary Table S8", pd.concat([rd("r5_equity.csv"), rd("r5_gaps.csv")])),
              ("Table S9", "Supplementary Table S9", pd.concat([rd("r4_whatif.csv"), rd("r6_targeted_beyond.csv")])),
              ("Table S10", "Supplementary Table S10", pd.concat([rd("r2_sensitivity.csv"), rd("r7_sensitivity.csv"), rd("r7_variance.csv")])),
              ("Figure S1", "Supplementary Figure S1", pd.json_normalize(json.load(open(OUT / "flow.json"))).T.reset_index().rename(columns={"index": "step", 0: "n"}).astype(str)),
              ("Figure S2A-B", "Supplementary Figure S2", binned_diffs()), ("Figure S2C", "Supplementary Figure S2", rd("r2_digits.csv")),
              ("Figure S3", "Supplementary Figure S3", pd.concat([rd("r2_cycles.csv"), rd("r3_trend.csv")])),
              ("Figure S4", "Supplementary Figure S4", rd("r3_calibration.csv"))]
    r = 3
    for name, key, df in sheets:
        ws = wb.create_sheet(name); ws["A1"] = cap[key]; ws["A1"].font = T
        df = df.loc[:, ~df.columns.duplicated()].fillna("")
        for j, c in enumerate(df.columns, 1):
            cell = ws.cell(3, j, str(c)); cell.font = XB; cell.fill = PatternFill("solid", start_color="E8ECF0"); cell.alignment = Alignment(wrap_text=True)
        for i, row in enumerate(df.itertuples(index=False), 4):
            for j, v in enumerate(row, 1):
                ws.cell(i, j, v.item() if hasattr(v, "item") else v).font = XF
        for j in range(1, len(df.columns) + 1):
            ws.column_dimensions[ws.cell(1, j).column_letter].width = 18
        toc.cell(r, 1, name).font = XB; toc.cell(r, 2, cap[key]).font = XF; r += 1
    toc.column_dimensions["A"].width = 22; toc.column_dimensions["B"].width = 100
    wb.save(path)


def main():
    dest = sys.argv[1] if len(sys.argv) > 1 else DEST
    os.makedirs(dest, exist_ok=True)
    m, s = items()
    for old in glob.glob(os.path.join(dest, "Figure_*.png")):     # 圖的組成改過，先清掉本程式之前產的舊圖
        os.remove(old)
    out = os.path.join(dest, "Tables_and_Figures.docx")
    print(book_base.build(m, out, s))
    for it in m + s:
        if it["kind"] == "figure":
            shutil.copy(it["path"], os.path.join(dest, it["path"].name)); print("  png ->", it["path"].name)
    source_data(os.path.join(dest, "Source_data.xlsx"), m, s)
    print("  xlsx -> Source_data.xlsx")


if __name__ == "__main__":
    main()
