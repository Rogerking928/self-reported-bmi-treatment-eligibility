"""圖（第二輪，外部意見 2026-10-23）：照 Bcc／氣喘那篇的樣式（DejaVu Sans、9 pt、無格線、分格字母粗體 13 pt 在軸外左上），
以印出寬度 180 mm 繪製，700 dpi PNG（不產 PDF）。存檔前用程式檢查文字框重疊與超出圖框（圖例壓到資料要另外看圖）。

主圖：1 機制（散布、P(自報合格 | 實測 BMI)、方向拆分；上列減重藥、下列手術）、2 誰被分錯（粗率 vs BMI 標準化）、3 怎麼辦（校正、灰色地帶）。
補充圖：S1 逐步篩選流程、S2 自報誤差與末位數、S3 各週期。
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "out"
W = 7.09
INK, AX, DARK, MID, LIGHT = "#1f2933", "#5b6b7a", "#2b333b", "#7d8893", "#c9d1d8"
# 彩色（Okabe–Ito，色盲安全）：顏色固定代表規則 —— 減重藥藍、手術朱紅；淺色為同色系的次要序列。其他區別仍用線型、實心／空心、形狀
COL = {"AOM": "#0072B2", "MBS": "#D55E00"}
PALE = {"AOM": "#7FB2D6", "MBS": "#EAA57A"}
CMAP = {"AOM": "Blues", "MBS": "Oranges"}
SEX = {"Women": "#CC79A7", "Men": "#009E73"}
plt.rcParams.update({"font.family": ["DejaVu Sans"], "font.size": 9, "axes.labelsize": 9, "xtick.labelsize": 8.5,
                     "ytick.labelsize": 8.5, "legend.fontsize": 8, "axes.edgecolor": AX, "axes.labelcolor": INK,
                     "xtick.color": AX, "ytick.color": AX, "text.color": INK, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": False, "axes.linewidth": 0.7, "legend.frameon": False})
RULE = {"AOM": "Anti-obesity medication", "MBS": "Metabolic and bariatric surgery"}
TH = {"AOM": (27, 30), "MBS": (30, 35)}
COND_LAB = {"AOM": ("Condition", "No condition"), "MBS": ("Condition", "No condition")}


def letter(ax, s, dx=-30):
    ax.annotate(s, xy=(0, 1), xycoords="axes fraction", xytext=(dx, 8), textcoords="offset points",
                fontsize=13, fontweight="bold", va="bottom", ha="left")


def check_overlap(fig):
    fig.canvas.draw(); r = fig.canvas.get_renderer()
    hidden = set()
    for ax in fig.axes:   # 視窗範圍外的刻度標籤不會畫出來，不列入檢查
        for axis, lim in ((ax.xaxis, ax.get_xlim()), (ax.yaxis, ax.get_ylim())):
            lo, hi = min(lim), max(lim)
            for tk in axis.get_major_ticks():
                if not (lo - 1e-9 <= tk.get_loc() <= hi + 1e-9):
                    hidden.update({id(tk.label1), id(tk.label2)})
    texts = [t for t in fig.findobj(matplotlib.text.Text) if t.get_visible() and t.get_text().strip() and id(t) not in hidden]
    boxes = [(t, t.get_window_extent(r)) for t in texts]
    bad = [(a.get_text()[:30], b.get_text()[:30]) for i, (a, ba) in enumerate(boxes) for b, bb in boxes[i + 1:] if ba.overlaps(bb)]
    fb = fig.bbox
    out = [t.get_text()[:30] for t, b in boxes if b.x0 < fb.x0 - 1 or b.x1 > fb.x1 + 1 or b.y0 < fb.y0 - 1 or b.y1 > fb.y1 + 1]
    assert not bad, f"overlapping text: {bad}"
    assert not out, f"text outside figure: {out}"


def save(fig, name):
    check_overlap(fig)
    fig.savefig(OUT / name, dpi=700)
    plt.close(fig)
    print("saved", name)


def fig1():
    sc = pd.read_csv(OUT / "r2_scatter.csv"); cu = pd.read_csv(OUT / "r2_curves.csv")
    q = json.load(open(OUT / "r2_summary.json"))["quadrants"]
    fig = plt.figure(figsize=(W, 5.3))
    gs = fig.add_gridspec(2, 3, left=0.075, right=0.985, bottom=0.08, top=0.93, wspace=0.42, hspace=0.55)
    for i, r in enumerate(("AOM", "MBS")):
        lo, hi = TH[r]
        # A／D：散布（加權六角格）
        a = fig.add_subplot(gs[i, 0])
        hb = a.hexbin(sc.bmi_meas.clip(18, 48), sc.bmi_self.clip(18, 48), C=sc.w / 1e3, reduce_C_function=np.sum, gridsize=45, extent=(18, 48, 18, 48),
                      bins="log", cmap=CMAP[r], mincnt=1, linewidths=0)
        cax = a.inset_axes([0.05, 0.44 if r == "AOM" else 0.62, 0.04, 0.2])   # 放在左上象限、避開門檻線
        cb = fig.colorbar(hb, cax=cax); cb.outline.set_linewidth(0.4); cb.ax.tick_params(labelsize=6, length=1.5, pad=1, which="both")
        cb.ax.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
        lo_, hi_ = hb.get_array().min(), hb.get_array().max()
        tks = [t for t in (1, 10, 100, 1000, 10000) if lo_ <= t <= hi_]
        cb.set_ticks(tks); cb.set_ticklabels([f"{t:,}" for t in tks])
        cax.set_title("Adults,\nthousands", fontsize=6, loc="left", pad=3)
        a.plot([18, 48], [18, 48], color=AX, lw=0.6, ls=":")
        a.axvline(hi, color=INK, lw=0.8); a.axhline(hi, color=INK, lw=0.8)
        a.set_xlim(18, 48); a.set_ylim(18, 48); a.set_aspect("equal")
        a.set_xticks(range(20, 49, 10)); a.set_yticks(range(20, 49, 10))
        a.set_xlabel("Measured BMI, kg/m²"); a.set_ylabel("Self-reported BMI, kg/m²")
        qq = q[r]
        # 兩個不一致象限：右下（實測合格、自報不合格）、左上（自報合格、實測不合格）；字放在角落，不壓資料
        a.text(47.5, 19.0, f"{qq['meas_ge_self_lt']:.1f}%\nat {hi}", ha="right", va="bottom", fontsize=7, color=INK, fontweight="bold")
        a.text(hi - 0.8, 47.5, f"{qq['meas_lt_self_ge']:.1f}%\nat {hi}", ha="right", va="top", fontsize=7, color=INK, fontweight="bold")
        letter(a, "AD"[i])
        # B／E：P(自報判合格 | 實測 BMI)，點線為理想的階梯
        b = fig.add_subplot(gs[i, 1])
        for cl, col, ls, lab in [("with condition", COL[r], "-", COND_LAB[r][0]), ("without condition", PALE[r], "--", COND_LAB[r][1])]:
            s = cu[(cu.rule == r) & (cu.condition == cl)]
            b.fill_between(s.bmi_mid, s.p_lo, s.p_hi, color=col, alpha=0.18, lw=0)
            b.plot(s.bmi_mid, s.p_self_eligible, color=col, lw=1.1, ls=ls, label=lab)
            t = lo if cl == "with condition" else hi
            b.plot([20, t, t, 40], [0, 0, 100, 100], color=col, lw=0.7, ls=":")
        b.set_xlim(20, 40); b.set_ylim(-2, 102); b.set_xticks(range(20, 41, 5))
        b.set_xlabel("Measured BMI, kg/m²"); b.set_ylabel("Eligible by self-report, %")
        # 直接標在線旁（圖例框會壓到曲線）
        xc, xn = (26.3, 32.2) if r == "AOM" else (29.0, 36.4)
        b.text(xc, 62, "Condition", fontsize=7, color=COL[r], ha="right", va="center")
        if r == "AOM":
            b.text(xn, 28, "No condition", fontsize=7, color=COL[r], ha="left", va="center")
        else:
            b.text(36.3, 12, "No\ncondition", fontsize=7, color=COL[r], ha="left", va="center")
        letter(b, "BE"[i])
        # C／F：方向拆分（上：實測合格而自報不合格；下：自報合格而實測不合格）＋ 人口密度帶
        c = fig.add_subplot(gs[i, 2])
        s = cu[(cu.rule == r) & (cu.condition == "all")]
        c.bar(s.bmi_mid, s.measured_only, width=0.45, color=COL[r], lw=0)
        c.bar(s.bmi_mid, -s.self_only, width=0.45, color=PALE[r], lw=0)
        c.axhline(0, color=AX, lw=0.6)
        c.set_xlim(20, 40); c.set_ylim(-24, 45); c.set_xticks(range(20, 41, 5))
        c.set_yticks([-10, 0, 20, 40]); c.set_yticklabels(["10", "0", "20", "40"])
        c.set_xlabel("Measured BMI, kg/m²"); c.set_ylabel("Adults in bin, %")
        c.text(20.5, 43, "Eligible by measured\nvalues only", ha="left", va="top", fontsize=7, color=COL[r])
        c.text(39.5, -9, "Eligible by\nself-report only", ha="right", va="top", fontsize=7, color=COL[r])
        dens = c.inset_axes([0, 0.0, 1, 0.045], transform=c.transAxes)
        dens.imshow(s.pop_share.values[None, :], aspect="auto", cmap="Greys", extent=(s.bmi_mid.min() - 0.25, s.bmi_mid.max() + 0.25, 0, 1),
                    vmin=0, vmax=s.pop_share.max() * 1.3)
        dens.set_xlim(20, 40); dens.axis("off")
        letter(c, "CF"[i])
        fig.text(0.075, [0.965, 0.475][i], RULE[r], fontsize=9.5, fontweight="bold", ha="left", va="bottom", color=COL[r])
    save(fig, "Figure_1.png")


def fig2():
    sg = pd.read_csv(OUT / "r2_subgroups.csv", keep_default_na=False)
    sg = sg[sg.group != "Survey cycle"]
    asm = json.load(open(OUT / "r3_summary.json"))["asian_mbs_specific"]
    order = sg[sg.rule == "AOM"][["group", "subgroup"]].values.tolist()
    fig = plt.figure(figsize=(W, 4.8))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.15], left=0.235, right=0.995, bottom=0.11, top=0.83, wspace=0.08)
    rows, y, heads, prev = [], 0, [], None
    for g, s in order:
        if g != prev and g != "All":
            y += 0.4; heads.append((g, y)); y += 1
        prev = g; rows.append((g, s, y)); y += 1
    axes = []
    for j, r in enumerate(("AOM", "MBS")):
        a = fig.add_subplot(gs[j]); axes.append(a)
        x = sg[sg.rule == r]
        for g, s, yy in rows:
            t = x[(x.group == g) & (x.subgroup == s)].iloc[0]
            asn = r == "MBS" and s == "Non-Hispanic Asian"     # 這一列多一個三角形：三個符號平均分開，不碰到下一列
            o1, o2 = (-0.26, 0.0) if asn else (-0.16, 0.16)
            a.errorbar(t.crude, yy + o1, xerr=[[t.crude - t.crude_lo], [t.crude_hi - t.crude]], fmt="o", color=COL[r], ms=3.3, lw=0.9)
            # 減重藥的條件列：標準化層別本身含條件狀態，標準化值依定義等於粗率 → 不畫
            if not (r == "AOM" and g == "Weight-related condition"):
                a.errorbar(t.standardized, yy + o2, xerr=[[t.standardized - t.std_lo], [t.std_hi - t.standardized]], fmt="s", color=COL[r], mfc="white", ms=3.3, lw=0.9)
            if asn:
                a.errorbar(asm["discordant"], yy + 0.26, xerr=[[asm["discordant"] - asm["lo"]], [asm["hi"] - asm["discordant"]]], fmt="^", color=COL[r], ms=3.5, lw=0.9)
        a.axvline(x[x.group == "All"].iloc[0].crude, color=LIGHT, lw=0.8, zorder=0)
        a.set_ylim(y, -0.8); a.set_xlim(0, 12); a.set_xticks(range(0, 13, 3)); a.set_yticks([])
        a.spines["left"].set_visible(False)
        a.set_xlabel("Discordant, % (95% CI)")
        a.set_title({"AOM": "Anti-obesity\nmedication", "MBS": "Metabolic and\nbariatric surgery"}[r], fontsize=8, color=COL[r], fontweight="bold", pad=4)
    a0 = axes[0]
    for g, s, yy in rows:
        a0.text(-0.04, yy, s if g != "All" else "All adults", transform=a0.get_yaxis_transform(), ha="right", va="center", fontsize=7.5,
                fontweight="bold" if g == "All" else "normal")
    for g, yy in heads:
        a0.text(-0.04, yy, g, transform=a0.get_yaxis_transform(), ha="right", va="center", fontsize=7.5, fontweight="bold", color=AX)
    tx = fig.add_subplot(gs[2]); tx.axis("off"); tx.set_ylim(y, -0.8); tx.set_xlim(0, 1)
    for xx, h in [(0.0, "n"), (0.27, "Medication,\ncrude / std"), (0.66, "Surgery,\ncrude / std")]:
        tx.text(xx, -0.9, h, fontsize=7, fontweight="bold", va="bottom", color=AX)
    for g, s, yy in rows:
        a_ = sg[(sg.rule == "AOM") & (sg.group == g) & (sg.subgroup == s)].iloc[0]; m_ = sg[(sg.rule == "MBS") & (sg.group == g) & (sg.subgroup == s)].iloc[0]
        tx.text(0.0, yy, f"{int(a_.n):,}", fontsize=7, va="center")
        tx.text(0.27, yy, f"{a_.crude:.1f} / " + ("—" if g == "Weight-related condition" else f"{a_.standardized:.1f}"), fontsize=7, va="center")
        tx.text(0.66, yy, f"{m_.crude:.1f} / {m_.standardized:.1f}", fontsize=7, va="center")
    h = [plt.Line2D([], [], marker="o", color=AX, ls="", ms=3.3), plt.Line2D([], [], marker="s", color=AX, mfc="white", ls="", ms=3.3),
         plt.Line2D([], [], marker="^", color=COL["MBS"], ls="", ms=3.5)]
    fig.legend(h, ["Crude", "Standardised to measured-BMI distribution", "Asian-specific surgery threshold (BMI ≥ 27.5)"],
               loc="upper left", bbox_to_anchor=(0.01, 1.0), ncol=3, fontsize=7, handletextpad=0.3, columnspacing=1.2)
    save(fig, "Figure_2.png")


def fig3():
    fr = pd.read_csv(OUT / "r4_frontier.csv"); eq = pd.read_csv(OUT / "r4_equity.csv")
    fig = plt.figure(figsize=(W, 3.6))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.05], left=0.07, right=0.99, bottom=0.31, top=0.88, wspace=0.42)
    ST = [("Asymmetric 1.5:1", LIGHT, "-", 0.8), ("Asymmetric 3:1", LIGHT, "-", 0.8), ("Symmetric ± k", MID, "--", 1.0),
          ("Probability-based (empirical)", DARK, ":", 1.2), ("Asymmetric 2:1", None, "-", 1.6)]
    for j, r in enumerate(("AOM", "MBS")):
        a = fig.add_subplot(gs[j])
        for st, col, ls, lw in ST:
            g = fr[(fr.rule == r) & (fr.strategy == st)].sort_values("measured_pct")
            a.plot(g.measured_pct, g.residual, color=col or COL[r], ls=ls, lw=lw, label=st if j == 0 else None)
            if st == "Asymmetric 2:1":
                a.fill_between(g.measured_pct, g.res_lo, g.res_hi, color=COL[r], alpha=0.18, lw=0)
                p = g[np.isclose(g.value, 1.0)].iloc[0]
                a.plot(p.measured_pct, p.residual, "D", color=COL[r], ms=4)
        a.set_xlim(0, 35); a.set_ylim(0, 9.5)
        a.set_xlabel("Adults measured, %"); a.set_ylabel("Residual discordant, %")
        a.set_title(RULE[r], fontsize=8, color=COL[r], fontweight="bold", pad=4)
        letter(a, "AB"[j])
    # C：公平性（不對稱規則 k = 1 前後，依族群）
    c = fig.add_subplot(gs[2])
    groups = ["All adults", "Non-Hispanic White", "Non-Hispanic Black", "Hispanic", "Non-Hispanic Asian", "Other or multiracial"]
    short = {"All adults": "All adults", "Non-Hispanic White": "White", "Non-Hispanic Black": "Black", "Hispanic": "Hispanic", "Non-Hispanic Asian": "Asian",
             "Other or multiracial": "Other or\nmultiracial"}
    for k, (r, mk, off) in enumerate([("AOM", "o", -0.15), ("MBS", "s", 0.15)]):
        for i, g in enumerate(groups):
            t = eq[(eq.rule == r) & (eq.group == g)].iloc[0]
            c.plot([t.residual_none, t.residual_k1], [i + off, i + off], color=PALE[r], lw=0.9, zorder=1)
            c.errorbar(t.residual_none, i + off, xerr=[[t.residual_none - t.res_none_lo], [t.res_none_hi - t.residual_none]], fmt=mk, color=COL[r], mfc="white",
                       ms=3.5, lw=0.6, zorder=2)
            c.errorbar(t.residual_k1, i + off, xerr=[[t.residual_k1 - t.res_k1_lo], [t.res_k1_hi - t.residual_k1]], fmt=mk, color=COL[r], ms=3.5, lw=0.8, zorder=3)
    c.set_yticks(range(len(groups))); c.set_yticklabels([short[g] for g in groups], fontsize=7.5); c.set_ylim(len(groups) - 0.4, -0.6)
    c.get_yticklabels()[0].set_fontweight("bold"); c.axhline(0.5, color=LIGHT, lw=0.6)
    c.set_xlim(0, 12); c.set_xlabel("Discordant, %"); c.set_title("By race and ethnicity", fontsize=8, color=AX, pad=4)
    letter(c, "C", dx=-38)
    h = [plt.Line2D([], [], color=AX, lw=1.6, marker="D", ms=3.5), plt.Line2D([], [], color=MID, ls="--"), plt.Line2D([], [], color=DARK, ls=":", lw=1.2),
         plt.Line2D([], [], color=LIGHT, lw=0.8)]
    fig.legend(h, ["Asymmetric 2:1, rule colour (◆ k = 1)", "Symmetric ± k", "Probability-based", "Asymmetric 1.5:1, 3:1"],
               loc="lower center", bbox_to_anchor=(0.36, 0.0), ncol=2, fontsize=7, handlelength=1.8, columnspacing=1.2)
    hc = [plt.Line2D([], [], marker="o", color=AX, mfc="white", ls="", ms=3.5), plt.Line2D([], [], marker="o", color=AX, ls="", ms=3.5),
          plt.Line2D([], [], marker="o", color=COL["AOM"], ls="", ms=3.5), plt.Line2D([], [], marker="s", color=COL["MBS"], ls="", ms=3.5)]
    fig.legend(hc, ["Self-report only", "Asymmetric rule, k = 1", "Medication", "Surgery"], loc="lower center", bbox_to_anchor=(0.84, 0.0),
               ncol=2, fontsize=7, handletextpad=0.3, columnspacing=1.0)
    save(fig, "Figure_3.png")


def fig_s4():
    cal = pd.read_csv(OUT / "r3_calibration.csv")
    fig, a = plt.subplots(figsize=(W * 0.55, 2.7), gridspec_kw=dict(left=0.15, right=0.97, bottom=0.17, top=0.82))
    for j, r in enumerate(("AOM", "MBS")):
        t = cal[cal.rule == r].iloc[0]
        a.errorbar(j - 0.15, t.self_reported, yerr=[[t.self_reported - t.self_lo], [t.self_hi - t.self_reported]], fmt="o", color=COL[r], ms=4, lw=0.9)
        a.errorbar(j + 0.15, t.corrected, yerr=[[t.corrected - t.cor_lo], [t.cor_hi - t.corrected]], fmt="s", color=COL[r], mfc="white", ms=4, lw=0.9)
        a.text(j, 11.8, f"Difference {t.difference:+.1f}\n({t.diff_lo:+.1f} to {t.diff_hi:+.1f})".replace("-", "−"), ha="center", va="top", fontsize=7)
    a.set_xticks([0, 1]); a.set_xticklabels(["Anti-obesity\nmedication", "Metabolic and\nbariatric surgery"], fontsize=7.5); a.set_xlim(-0.6, 1.6); a.set_ylim(0, 12)
    a.set_ylabel("Discordant, % (95% CI)")
    h = [plt.Line2D([], [], marker="o", color=AX, ls="", ms=4), plt.Line2D([], [], marker="s", color=AX, mfc="white", ls="", ms=4)]
    fig.legend(h, ["Self-reported", "Corrected self-reported"], loc="upper center", bbox_to_anchor=(0.56, 1.0), ncol=2, fontsize=7.2)
    save(fig, "Figure_S4.png")


def fig_s1():
    f = json.load(open(OUT / "flow.json"))
    steps = [("Examined in the mobile examination centre, all ages", None),
             ("Adults aged 18 years or older", ("Aged < 18 years", f["examined (all ages)"] - f["adults >= 18 examined"])),
             ("Not pregnant", ("Pregnant", f["excluded: pregnant"])),
             ("Measured height and weight available", ("Measured height or weight missing", f["excluded: measured height or weight missing"])),
             ("No examiner comment on the measurement", ("Measurement flagged by examiner", f["excluded: measurement comment (clothing, appliance, other)"])),
             ("Analysed: self-reported height and weight available",
              ("Self-reported height or weight missing, refused,\ndon't know or implausible", f["excluded: self-reported height or weight missing, refused, don't know or implausible"]))]
    n = f["examined (all ages)"]; rows = []
    for t, ex in steps:
        if ex:
            n -= ex[1]
        rows.append((t, n, ex))
    assert rows[-1][1] == f["analysed"], (rows[-1][1], f["analysed"])
    fig, a = plt.subplots(figsize=(W, 6.0)); a.axis("off"); a.set_xlim(0, 10); a.set_ylim(0, 12.2)
    plt.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
    ys = np.linspace(11.4, 1.6, len(rows))
    for k, ((t, nn, ex), y) in enumerate(zip(rows, ys)):
        a.add_patch(FancyBboxPatch((0.2, y - 0.42), 4.8, 0.84, boxstyle="round,pad=0.02", fc="white", ec=AX, lw=0.8))
        a.text(2.6, y, f"{t}\nn = {nn:,}", ha="center", va="center", fontsize=7.5)
        if k:
            a.annotate("", xy=(2.6, y + 0.42), xytext=(2.6, ys[k - 1] - 0.42), arrowprops=dict(arrowstyle="-|>", color=AX, lw=0.8))
            ym = (y + ys[k - 1]) / 2
            a.annotate("", xy=(5.6, ym), xytext=(2.6, ym), arrowprops=dict(arrowstyle="-|>", color=AX, lw=0.7))
            a.add_patch(FancyBboxPatch((5.6, ym - 0.36), 4.25, 0.72, boxstyle="round,pad=0.02", fc="white", ec=AX, lw=0.7, ls="--"))
            a.text(5.75, ym, f"Excluded: {ex[0]}\nn = {ex[1]:,}", ha="left", va="center", fontsize=7)
    cyc = f["analysed by cycle"]
    a.text(5.0, 0.55, "Analysed by cycle: " + "; ".join(f"{k.replace('-', '–')} {v:,}" for k, v in cyc.items()), ha="center", fontsize=7.5, color=AX)
    save(fig, "Figure_S1.png")


def fig_s2():
    d = pd.read_csv(OUT / "differences.csv"); g = pd.read_csv(OUT / "r2_digits.csv")
    fig, ax = plt.subplots(1, 3, figsize=(W, 3.0), gridspec_kw=dict(left=0.08, right=0.99, bottom=0.25, top=0.88, wspace=0.42))
    for a, col, lab, rng_ in [(ax[0], "weight_diff_kg", "Self-reported minus measured\nweight, kg", (-15, 15)),
                              (ax[1], "height_diff_cm", "Self-reported minus measured\nheight, cm", (-10, 10))]:
        bins = np.linspace(*rng_, 61)
        for sx, c_, ls in [("Women", SEX["Women"], "-"), ("Men", SEX["Men"], "--")]:
            s = d[d.sex == sx]
            h, e = np.histogram(s[col].clip(*rng_), bins=bins, weights=s.w)
            a.step(e[:-1], 100 * h / h.sum(), where="post", color=c_, lw=1.0, ls=ls, label=sx)
        a.axvline(0, color=LIGHT, lw=0.8, zorder=0)
        a.set_xlabel(lab); a.set_ylabel("Adults, %"); a.set_xlim(*rng_); a.set_ylim(bottom=0)
        tk = list(np.arange(rng_[0], rng_[1] + 1, 5)); a.set_xticks(tk)
        a.set_xticklabels([(f"≤{t}" if t == rng_[0] else f"≥{t}" if t == rng_[1] else f"{t}").replace("-", "−") for t in tk], fontsize=7.5)
    ax[0].legend(loc="upper left", bbox_to_anchor=(-0.02, 1.0), handlelength=1.4, fontsize=7.5)
    x = np.arange(10)
    ax[2].bar(x - 0.2, g.weight_pct, width=0.38, color=DARK, label="Weight, pounds")
    ax[2].bar(x + 0.2, g.height_pct, width=0.38, color=LIGHT, label="Height, inches")
    ax[2].axhline(10, color=MID, lw=0.7, ls="--")
    ax[2].set_xticks(range(10)); ax[2].set_ylim(0, 50)
    ax[2].set_xlabel("Last digit of\nself-reported value"); ax[2].set_ylabel("Adults, %")
    ax[2].legend(loc="upper right", fontsize=7.5, handlelength=1.2)
    for a, s in zip(ax, "ABC"):
        letter(a, s)
    save(fig, "Figure_S2.png")


def fig_s3():
    cy = pd.read_csv(OUT / "r2_cycles.csv"); trd = pd.read_csv(OUT / "r3_trend.csv").set_index("rule")
    MIDY = {"2013-2014": 2014, "2015-2016": 2016, "2017-2018": 2018, "2021-2023": 2022.6}   # 2021–23 中點約 2022.6
    fig, a = plt.subplots(figsize=(W * 0.62, 2.8), gridspec_kw=dict(left=0.13, right=0.97, bottom=0.2, top=0.8))
    a.axvspan(2019, 2021, color=LIGHT, alpha=0.35, lw=0)
    a.text(2020, 0.5, "2019–2020 cycle\nnot completed\n(COVID-19)", ha="center", va="bottom", fontsize=6.5, color=AX)
    for k, (r, col, mk) in enumerate([("AOM", COL["AOM"], "o"), ("MBS", COL["MBS"], "s")]):
        s = cy[cy.rule == r]
        xs = s.cycle.map(MIDY).values + (k - 0.5) * 0.35
        a.errorbar(xs, s.discordant, yerr=[s.discordant - s.lo, s.hi - s.discordant], fmt=mk, color=col, mfc=col if k == 0 else "white", ms=3.5, lw=0.9,
                   label=f"{RULE[r]}: {trd.loc[r, 'slope_pp_per_10y']:+.1f} ({trd.loc[r, 'lo']:+.1f} to {trd.loc[r, 'hi']:+.1f}) pp per 10 years".replace("-", "−").replace("−obesity", "-obesity"))
    a.set_xticks(list(MIDY.values())); a.set_xticklabels([c.replace("-", "–") for c in MIDY], fontsize=7.5); a.set_xlim(2012.8, 2023.2)
    a.set_ylim(0, 10); a.set_ylabel("Discordant, % (95% CI)"); a.set_xlabel("NHANES cycle")
    a.legend(loc="lower left", bbox_to_anchor=(-0.02, 1.0), ncol=1, fontsize=6.8, handletextpad=0.3)
    save(fig, "Figure_S3.png")


if __name__ == "__main__":
    fig1(); fig2(); fig3(); fig_s1(); fig_s2(); fig_s3(); fig_s4()
