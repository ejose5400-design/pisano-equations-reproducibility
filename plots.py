"""Paper Figures 1--5 and Appendix Figure A1; procedural plotting functions."""

from pathlib import Path
import numpy as np
import matplotlib

matplotlib.use("Agg")  # Headless rendering: no desktop window is required.
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from model import NAMES, LABELS, BOUNDS, ALPHA, COEFF, uniform_log_variance


def figures(x, y, sens, pairs, out):
    """Regenerate Figures 1--5 using exactly the manuscript's numerical inputs.

    The plotting recipe preserves the supplied dimensions, palette, bins, fonts,
    600-dpi raster exports, vector text/lines, and contour-label positions.
    Figures 4 and 5 use a blue histogram, orange P50, red P95, and an upper-left
    legend. Response surfaces hold the other inputs at geometric midpoints.
    The code uses functions rather than a user-defined class hierarchy; plotting
    and LHS library handles are standard Matplotlib/SciPy interfaces."""
    from matplotlib.ticker import MaxNLocator

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 13,
            "axes.labelsize": 13,
            "axes.titlesize": 14,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 12,
            "axes.linewidth": 0.8,
            "axes.spines.top": True,
            "axes.spines.right": True,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.facecolor": "white",
        }
    )
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    blue = colors[0]

    def save(fig, stem):
        for ext in ["pdf", "svg", "png"]:
            target = out / (stem + "." + ext)
            temporary = out / (stem + "." + ext + ".tmp")
            fig.savefig(
                temporary,
                format=ext,
                dpi=600,
                facecolor="white",
                bbox_inches="tight",
                pad_inches=0.04,
            )
            temporary.replace(target)
        plt.close(fig)

    def grid(ax, axis="both"):
        ax.set_axisbelow(True)
        ax.grid(True, axis=axis, color="#e6e6e6", linewidth=0.7)

    # Figure 1: baseline induced distributions and same-row log ratios.
    fig, ax = plt.subplots(1, 2, figsize=(7.4, 3.35), layout="constrained")
    for k in range(4):
        ax[0].hist(
            y[:, k],
            bins=150,
            density=True,
            histtype="step",
            lw=1.25,
            label=f"EQ{k + 1}",
            color=colors[k],
        )
    ax[0].set(
        xlabel="$\\log_{10}(M_{\\mathrm{SpTS}}\\,[\\mathrm{kg\\,day^{-1}}])$",
        ylabel="Probability density",
        title="A. Output distributions",
    )
    ax[0].legend(frameon=False)
    grid(ax[0])
    data = [y[:, i] - y[:, j] for i in range(4) for j in range(i + 1, 4)]
    ax[1].boxplot(
        data,
        showfliers=False,
        tick_labels=[f"EQ{i + 1}/EQ{j + 1}" for i in range(4) for j in range(i + 1, 4)],
        medianprops=dict(color=colors[1], linewidth=1.4),
        boxprops=dict(linewidth=1),
        whiskerprops=dict(linewidth=1),
        capprops=dict(linewidth=1),
    )
    ax[1].tick_params(axis="x", rotation=40)
    ax[1].axhline(0, color=blue, lw=0.8)
    ax[1].set(ylabel="$\\log_{10}(M_i/M_j)$", title="B. Pairwise disagreement")
    grid(ax[1])
    save(fig, "global_comparison")
    # Figure 2: rank association is distinct from agreement in magnitude.
    corr = spearmanr(y).statistic
    fig, ax = plt.subplots(figsize=(4.7, 4.1), layout="constrained")
    im = ax.imshow(corr, vmin=0.64, vmax=1, cmap="viridis")
    ax.set(
        xticks=range(4),
        yticks=range(4),
        xticklabels=[f"EQ{i + 1}" for i in range(4)],
        yticklabels=[f"EQ{i + 1}" for i in range(4)],
        title="Agreement between equation outputs",
    )
    ax.set_xticks(np.arange(-0.5, 4, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, 4, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.1)
    ax.tick_params(which="minor", length=0)
    for i in range(4):
        for j in range(4):
            ax.text(
                j,
                i,
                f"{corr[i, j]:.3f}",
                ha="center",
                va="center",
                color="black" if corr[i, j] > 0.9 else "white",
                fontweight="bold" if i != j and corr[i, j] > 0.9 else "normal",
                fontsize=11,
            )
    fig.colorbar(
        im,
        ax=ax,
        label="Spearman correlation",
        ticks=[0.65, 0.75, 0.85, 0.95, 1],
        fraction=0.048,
        pad=0.045,
    )
    save(fig, "rank_agreement")
    # Figure 3: exact independent population shares plus sampled PRCCs.
    var = uniform_log_variance(*BOUNDS.T)
    sh = ALPHA**2 * var
    sh /= sh.sum(axis=1)[:, None]
    pr = np.full((4, 7), np.nan)
    for row in sens.itertuples():
        pr[row.equation - 1, NAMES.index(row.input)] = row.PRCC
    order = np.array([0, 4, 2, 6, 1, 3, 5])
    pos = np.arange(7)
    fig, axs = plt.subplots(
        1,
        2,
        figsize=(7.4, 3.9),
        layout="constrained",
        gridspec_kw={"width_ratios": [1, 1.1]},
    )
    axs[0].barh(
        pos + 0.17,
        np.max(sh[:, order], axis=0),
        height=0.32,
        color=blue,
        label="Max exact log-share",
    )
    axs[0].barh(
        pos - 0.17,
        np.nanmax(abs(pr[:, order]), axis=0),
        height=0.32,
        color=colors[1],
        label="Max $|\\mathrm{PRCC}|$",
    )
    axs[0].set(
        yticks=pos,
        yticklabels=[LABELS[v] for v in order],
        xlim=(0, 1.02),
        xlabel="Maximum across equations",
        title="A. Global variable screening",
    )
    axs[0].invert_yaxis()
    axs[0].legend(frameon=False, loc="lower right", fontsize=10)
    grid(axs[0], "x")
    cm = plt.get_cmap("RdBu_r").copy()
    cm.set_bad("#f2f2f2")
    arr = pr[:, order].T
    im = axs[1].imshow(arr, cmap=cm, vmin=-1, vmax=1, aspect="auto")
    axs[1].set(
        xticks=range(4),
        yticks=range(7),
        xticklabels=[f"EQ{k + 1}" for k in range(4)],
        yticklabels=[LABELS[v] for v in order],
        title="B. Signed PRCC structure",
    )
    axs[1].set_xticks(np.arange(-0.5, 4, 1), minor=True)
    axs[1].set_yticks(np.arange(-0.5, 7, 1), minor=True)
    axs[1].grid(which="minor", color="white", lw=1)
    axs[1].tick_params(which="minor", length=0)
    for i in range(7):
        for j in range(4):
            if np.isfinite(arr[i, j]):
                axs[1].text(
                    j,
                    i,
                    f"{arr[i, j]:.2f}",
                    ha="center",
                    va="center",
                    fontsize=10,
                    color="white" if abs(arr[i, j]) > 0.45 else "black",
                    fontweight="bold" if abs(arr[i, j]) > 0.65 else "normal",
                )
    fig.colorbar(im, ax=axs[1], label="PRCC", fraction=0.045, pad=0.045, shrink=0.92)
    save(fig, "sensitivity")
    # Figures 4 and 5: identical four-panel recipe for EQ1 and EQ4.
    for k, v in [(0, 4), (3, 2)]:
        use = np.flatnonzero(ALPHA[k])
        effects = ALPHA[k, use] * np.log10(BOUNDS[use, 1] / BOUNDS[use, 0])
        use = use[np.argsort(-effects)]
        effects = ALPHA[k, use] * np.log10(BOUNDS[use, 1] / BOUNDS[use, 0])
        s = sens[sens.equation == k + 1].set_index("input")
        pos = np.arange(len(use))
        fig, ax = plt.subplots(2, 2, figsize=(7.4, 6.1), layout="constrained")
        qs = np.quantile(y[:, k], [0.05, 0.5, 0.95])
        ax[0, 0].hist(
            y[:, k],
            bins=140,
            density=True,
            color=blue,
            alpha=0.85,
            histtype="stepfilled",
            edgecolor="none",
        )
        for q, name, line_color in zip(qs[1:], ["P50", "P95"], [colors[1], colors[3]]):
            ax[0, 0].axvline(q, color=line_color, ls="--", lw=1.5, label=name)
        ax[0, 0].set(
            xlabel="$\\log_{10}(M_{\\mathrm{SpTS}}\\,[\\mathrm{kg\\,day^{-1}}])$",
            ylabel="Density",
            title="A. Output distribution",
            xlim=(-1, 4),
        )
        ax[0, 0].legend(frameon=False, loc="upper left")
        grid(ax[0, 0])
        ax[0, 1].barh(pos, effects, color=blue, height=0.72)
        ax[0, 1].set(
            yticks=pos,
            yticklabels=[LABELS[i] for i in use],
            xlabel="Orders of magnitude",
            title="B. Range effect",
        )
        ax[0, 1].invert_yaxis()
        ax[0, 1].axvline(0, color=blue, lw=0.9)
        grid(ax[0, 1], "x")
        prof = np.array([s.loc[NAMES[i], "U"] for i in use])
        ax[1, 0].barh(pos, prof, color=blue, height=0.72)
        ax[1, 0].set(
            yticks=pos,
            yticklabels=[LABELS[i] for i in use],
            xlabel="$\\log_2(\\mathrm{top\\ 5\\%}/\\mathrm{rest})$",
            title="C. Upper-tail profile",
        )
        ax[1, 0].invert_yaxis()
        ax[1, 0].axvline(0, color=blue, lw=0.9)
        grid(ax[1, 0], "x")
        lg = np.linspace(*np.log10(BOUNDS[0]), 500)
        sg = np.linspace(*np.log10(BOUNDS[v]), 500)
        ll, ss = np.meshgrid(lg, sg)
        midpoint = np.sqrt(BOUNDS.prod(axis=1))
        constant = np.log10(COEFF[k]) + sum(
            (ALPHA[k, i] * np.log10(midpoint[i]) for i in use if i not in [0, v])
        )
        z = constant + ALPHA[k, 0] * ll + ALPHA[k, v] * ss
        # Raster colour field avoids vector mesh seams; labels remain vector.
        im = ax[1, 1].imshow(
            z,
            origin="lower",
            extent=[lg[0], lg[-1], sg[0], sg[-1]],
            aspect="auto",
            cmap="viridis",
            interpolation="bilinear",
            rasterized=True,
        )
        cs = ax[1, 1].contour(lg, sg, z, levels=qs, colors="black", linewidths=1.05)
        manual_points = []
        for idx, (q, name) in enumerate(zip(qs, ["P5", "P50", "P95"])):
            segs = [seg for seg in cs.allsegs[idx] if len(seg) > 1]
            if segs:
                seg = max(segs, key=len)
                point = seg[len(seg) // 2]
                manual_points.append(point)
        labels = ax[1, 1].clabel(
            cs,
            fmt=dict(zip(qs, ["P5", "P50", "P95"])),
            manual=manual_points,
            inline=True,
            inline_spacing=4,
            fontsize=12,
        )
        import matplotlib.patheffects as pe

        for label in labels:
            label.set_path_effects([pe.withStroke(linewidth=1.8, foreground="white")])
        ax[1, 1].set(
            xlabel="$\\log_{10}(L\\,[\\mathrm{m}])$",
            ylabel="$\\log_{10}(S_{PD})$" if k == 0 else "$\\log_{10}(S)$",
            title="D. Response surface",
        )
        ax[1, 1].xaxis.set_major_locator(MaxNLocator(5))
        ax[1, 1].yaxis.set_major_locator(MaxNLocator(5))
        fig.colorbar(
            im,
            ax=ax[1, 1],
            label="$\\log_{10}(M_{\\mathrm{SpTS}})$",
            fraction=0.048,
            pad=0.045,
        )
        save(fig, f"eq{k + 1}_diagnostics")


def slope_schematic(out):
    """Draw Appendix Figure A1: model-derived p and geometry-derived quantiles.

    The exponential CDF, mean 0.012 and p=0.65 are illustrative only. The plot
    does not contain measured sewer data or reproduce endogenous PLD. The
    inequality SPD4 <= SPD follows from evaluating one monotone quantile curve."""
    from matplotlib.patches import FancyBboxPatch

    # Match a standalone diagram run independently of the preceding paper plots.
    plt.rcdefaults()

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    )
    fig = plt.figure(figsize=(10, 5.2), layout="constrained")
    gs = fig.add_gridspec(1, 2, width_ratios=[1.12, 1])
    ax = fig.add_subplot(gs[0, 0])
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.axis("off")

    def box(y, text, color):
        p = FancyBboxPatch(
            (0.04, y),
            0.92,
            0.16,
            boxstyle="round,pad=0.015",
            facecolor="white",
            edgecolor=color,
            linewidth=1.4,
        )
        ax.add_patch(p)
        ax.text(0.5, y + 0.08, text, ha="center", va="center", fontsize=11)

    box(
        0.78,
        "Prior deposition-model output\nCumulative deposited mass vs. pipe length",
        "#1f77b4",
    )
    box(
        0.51,
        "Read length fraction at 80% of total mass\n" + "$p=P_{LD}/100$; $L_{PD}=pL$",
        "#1f77b4",
    )
    box(
        0.24,
        "Network pipe slopes and segment lengths\nLength-weighted cumulative distribution "
        + "$F_S(s)$",
        "#ff7f0e",
    )
    ax.annotate(
        "",
        xy=(0.5, 0.69),
        xytext=(0.5, 0.78),
        arrowprops={"arrowstyle": "->", "lw": 1.3},
    )
    ax.text(
        0.5,
        0.445,
        "Combine model-derived p with geometry-derived CDF",
        ha="center",
        fontsize=10,
    )
    ax.text(
        0.5,
        0.09,
        "$S_{PD}=F_S^{-1}(p)$;  $S_{PD/4}=F_S^{-1}(p/4)$",
        ha="center",
        fontsize=12,
    )
    ax.annotate(
        "",
        xy=(0.5, 0.14),
        xytext=(0.5, 0.24),
        arrowprops={"arrowstyle": "->", "lw": 1.3},
    )
    ax.set_title("A. Information required by EQ1", loc="left", fontsize=13, pad=10)
    ax2 = fig.add_subplot(gs[0, 1])
    s = np.linspace(0, 0.055, 500)
    mean = 0.012
    f = 1 - np.exp(-s / mean)
    p = 0.65
    ax2.plot(s, f, color="#1f77b4", lw=2)
    for prob, label, c in [
        (p, "$S_{PD}$", "#d62728"),
        (p / 4, "$S_{PD/4}$", "#ff7f0e"),
    ]:
        q = -mean * np.log1p(-prob)
        ax2.plot([0, q, q], [prob, prob, 0], ls="--", lw=1.3, color=c)
        ax2.annotate(
            label,
            xy=(q, 0),
            xytext=(q, -0.12),
            ha="center",
            annotation_clip=False,
            fontsize=12,
            color=c,
        )
    ax2.set(
        xlim=(0, 0.055),
        ylim=(0, 1),
        xlabel="Pipe slope, s",
        ylabel="Length fraction, $F_S(s)$",
        yticks=[0, p / 4, p, 1],
        yticklabels=["0", "$p/4$", "$p$", "1"],
    )
    ax2.xaxis.labelpad = 30
    ax2.grid(alpha=0.18)
    ax2.set_title(
        "B. Quantile lookup (illustrative CDF)", loc="left", fontsize=13, pad=10
    )
    for ext in ["pdf", "svg", "png"]:
        target = out / ("Figure_A1_EQ1_descriptors." + ext)
        tmp = target.with_suffix("." + ext + ".tmp")
        fig.savefig(tmp, format=ext, dpi=600, bbox_inches="tight", pad_inches=0.08)
        tmp.replace(target)
    plt.close(fig)
