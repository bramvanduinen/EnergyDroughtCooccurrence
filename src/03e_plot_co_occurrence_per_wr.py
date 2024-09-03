# %%

import pickle

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from config import PATH_DATA, REGIONS, VERSION
from matplotlib import cm, colors, gridspec
from matplotlib.patches import Rectangle

# %% load data
co_occurrence_dict = pickle.load(open(f"{PATH_DATA}/{VERSION}/wr_meanprobs_absolute.pickle", "rb"))
co_occurrence_all = np.load(f"{PATH_DATA}/{VERSION}/co_occurrence_regions_all.npy")

wrs = np.arange(6)
REGION_LIST = list(REGIONS.keys())
region_order = ["IB", "C", "E", "NW", "N", "B"]  # hardcoded, copied from from 03c run
row_order = [0, 4, 5, 1, 2, 3]  # hardcoded, copied from from 03c run

BOUNDS_GREY = np.arange(-0.3, 0.35, 0.05)
CMAP_GREY = cm.get_cmap("RdBu_r", len(BOUNDS_GREY) - 1)
NORM_GREY = colors.BoundaryNorm(BOUNDS_GREY, CMAP_GREY.N)
CMAP_GREY.set_bad((0.95, 0.95, 0.95))
tick_vals = np.arange(-0.3, 0.35, 0.05).round(2)
vmin_diff, vmax_diff = -0.3, 0.3
extend = "max"

bounds = np.arange(0, 1.1, 0.1)
cmap = colors.ListedColormap(sns.color_palette("Reds", len(bounds) - 1))
norm_cmap = colors.BoundaryNorm(bounds, cmap.N)

fontsize = 14


# %%
def plot_risk_ratios(ax_bar, regime):
    risk_ratio = co_occurrence_dict[regime]["risk_ratios"]

    num_heatmap_rows = len(REGIONS)

    # risk ratio bar plot
    bar_widths = np.abs(risk_ratio - 1)[row_order]
    bar_starts = np.where(risk_ratio < 1, risk_ratio, 1)[row_order]

    for j, country in enumerate(np.array(REGION_LIST)[row_order]):
        left = bar_starts[j]
        width = bar_widths[j]
        ax_bar.barh(
            y=country,
            width=width,
            height=0.2,
            left=left,
            color="red",
            edgecolor="black",
        )

    ax_bar.set_ylim(-0.5, num_heatmap_rows - 0.5)
    ax_bar.invert_yaxis()
    ax_bar.set(yticklabels=[], yticks=[])
    ax_bar.set_xscale("log")
    if regime == "NAO +":
        ax_bar.set_xlim(0.01, 2)
        ax_bar.set_xticks([0.1, 0.5, 1], [0.1, 0.5, 1], size=fontsize - 2, rotation=90)
    else:
        ax_bar.set_xlim(0.1, 5)
        ax_bar.set_xticks([0.2, 0.5, 1, 2, 5], [0.2, 0.5, 1, 2, 5], size=fontsize - 2, rotation=90)
    ax_bar.set_xlabel(" Risk ratio", size=fontsize)
    xaxis_line = ax_bar.get_xaxis().get_gridlines()[0]
    line_width = xaxis_line.get_linewidth()
    ax_bar.axvline(x=1, color="k", linestyle="-", linewidth=line_width)


def plot_heatmap(ax_heatmap, regime, type="diff"):
    # ax_heatmap = fig.add_subplot(gs_heatmap)

    meanprob_all = co_occurrence_dict[regime]["mean_all"]
    meanprob_wr = co_occurrence_dict[regime]["mean"]
    low_snr_indices = co_occurrence_dict[regime]["low_snr_indices"]
    high_pval_indices = co_occurrence_dict[regime]["high_pval_indices"]

    if type == "diff":
        diff = meanprob_wr - meanprob_all
        np.fill_diagonal(diff, np.nan)
        cmap_plot = CMAP_GREY
        vmin, vmax = vmin_diff, vmax_diff
    elif type == "all":
        diff = meanprob_wr
        np.fill_diagonal(diff, 1)
        cmap_plot = cmap
        vmin, vmax = 0, 1

    cg = sns.heatmap(
        diff,
        vmin=vmin,
        vmax=vmax,
        cmap=cmap_plot,
        cbar=False,
        xticklabels=region_order,
        yticklabels=region_order,
        ax=ax_heatmap,
    )
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize, rotation=0)

    ax = cg.axes
    for k, l in zip(*low_snr_indices):
        ax.text(l + 0.5, k + 0.625, "*", ha="center", va="center", color="k", fontsize=25)

    if type == "diff":
        for k, l in zip(*high_pval_indices):
            rect = Rectangle(
                (l, k),
                1,
                1,
                linewidth=0,
                edgecolor="k",
                facecolor="none",
                hatch="//",
                zorder=2,
            )
            ax.add_patch(rect)
    return cg


# %%
plt.figure(figsize=(8, 6), dpi=300)
cg = sns.heatmap(
    co_occurrence_all,
    cmap=cmap,
    norm=norm_cmap,
    xticklabels=region_order,
    yticklabels=region_order,
    cbar=False,
)
cg.set_xticklabels(cg.get_xmajorticklabels(), fontsize=fontsize)
cg.set_yticklabels(cg.get_ymajorticklabels(), fontsize=fontsize, rotation=0)

cbar = plt.colorbar(cg.collections[0])
cbar.ax.tick_params(labelsize=fontsize + 2)
cbar.set_ticks(bounds)
cbar.set_label(r"P($E_{r,column}$|$E_{r,row}$)", size=fontsize + 4)
plt.title("(a) All regimes", size=fontsize + 4, loc="left")

fig = plt.figure(figsize=(8, 6), dpi=300)
hm_top_right = plot_heatmap(plt.gca(), "NAO -")
plt.title("(b) NAO \u2212", size=fontsize + 4, loc="left")
cbar_diff = plt.colorbar(hm_top_right.collections[0], extend="max")
cbar_diff.ax.tick_params(labelsize=fontsize + 2)
cbar_diff.set_label(r"Absolute probability difference [-]", size=fontsize + 4)

plt.figure(figsize=(7, 6.5), dpi=300)
plot_heatmap(plt.gca(), "Blocking")
plt.title("(c) Blocking", size=fontsize + 4, loc="left")

plt.figure(figsize=(7, 6.5), dpi=300)
plot_heatmap(plt.gca(), "Atl. Ridge")
plt.title("(d) Atl. Ridge", size=fontsize + 4, loc="left")

# %% Appendix version, with the other regimes

fig = plt.figure(figsize=(8, 6), dpi=300)
hm_top_right = plot_heatmap(plt.gca(), "NAO +")
plt.title("(b) NAO +", size=fontsize + 4, loc="left")
cbar_diff = plt.colorbar(hm_top_right.collections[0], extend="min")
cbar_diff.ax.tick_params(labelsize=fontsize + 2)
cbar_diff.set_label(r"Absolute probability difference [-]", size=fontsize + 4)

plt.figure(figsize=(7, 6.5), dpi=300)
plot_heatmap(plt.gca(), "No regime")
plt.title("(c) No regime", size=fontsize + 4, loc="left")

plt.figure(figsize=(7, 6.5), dpi=300)
plot_heatmap(plt.gca(), "No persisting regime")
plt.title("(d) No persisting regime", size=fontsize + 4, loc="left")

# %% Appendix version, with all absolute probabilities
plt.figure(figsize=(7, 6.5), dpi=300)
plot_heatmap(plt.gca(), "NAO +", type="all")
plt.title("(a) NAO +", size=fontsize + 4, loc="left")

plt.figure(figsize=(7, 6.5), dpi=300)
plot_heatmap(plt.gca(), "NAO -", type="all")
plt.title("(b) NAO \u2212", size=fontsize + 4, loc="left")

plt.figure(figsize=(7, 6.5), dpi=300)
plot_heatmap(plt.gca(), "Blocking", type="all")
plt.title("(c) Blocking", size=fontsize + 4, loc="left")

plt.figure(figsize=(7, 6.5), dpi=300)
plot_heatmap(plt.gca(), "Atl. Ridge", type="all")
plt.title("(d) Atl. Ridge", size=fontsize + 4, loc="left")

plt.figure(figsize=(7, 6.5), dpi=300)
plot_heatmap(plt.gca(), "No regime", type="all")
plt.title("(e) No regime", size=fontsize + 4, loc="left")

plt.figure(figsize=(7, 6.5), dpi=300)
plot_heatmap(plt.gca(), "No persisting regime", type="all")
plt.title("(f) No persisting regime", size=fontsize + 4, loc="left")

# %% trying to plot it all at once, but didn't turn out nicely

fig = plt.figure(figsize=(15, 9), dpi=300)
# gs = gridspec.GridSpec(2, 2, height_ratios=[1, 1], width_ratios=[1, 1])
gs = gridspec.GridSpec(2, 3, height_ratios=[1, 1], width_ratios=[1, 1, 0.05])
plt.subplots_adjust(wspace=0.2, hspace=0.3)
ax_heatmap_top_left = plt.subplot(gs[0, 0])
cg = sns.heatmap(
    co_occurrence_all,
    cmap=cmap,
    norm=norm_cmap,
    xticklabels=region_order,
    yticklabels=region_order,
    cbar=False,
)
cg.set_xticklabels(cg.get_xmajorticklabels(), fontsize=fontsize)
cg.set_yticklabels(cg.get_ymajorticklabels(), fontsize=fontsize, rotation=0)

cbar = plt.colorbar(cg.collections[0])
cbar.ax.tick_params(labelsize=16)
cbar.set_ticks(bounds)
cbar.set_label(r"P($E_{r,column}$|$E_{r,row}$)", size=fontsize)
plt.title("(a) All regimes", size=fontsize + 4, loc="left")

hm_top_right = plot_heatmap(gs[0, 1], "NAO -")
gs[0, 1].set_title("     (b) NAO -", size=fontsize + 4)

gs_bottom_left = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[1, 0], width_ratios=[1, 6])
ax_bar_bottom_left = plt.subplot(gs_bottom_left[0])
ax_heatmap_bottom_left = plt.subplot(gs_bottom_left[1])
plot_risk_ratios(ax_bar_bottom_left, "Blocking")
plot_heatmap(ax_heatmap_bottom_left, "Blocking")
ax_bar_bottom_left.set_title("         (c) Blocking", size=fontsize + 4)

gs_bottom_right = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[1, 1], width_ratios=[1, 6])
ax_bar_bottom_right = plt.subplot(gs_bottom_right[0])
ax_heatmap_bottom_right = plt.subplot(gs_bottom_right[1])
plot_risk_ratios(ax_bar_bottom_right, "Atl. Ridge")
plot_heatmap(ax_heatmap_bottom_right, "Atl. Ridge")
ax_bar_bottom_right.set_title("           (d) Atl. Ridge", size=fontsize + 4)

cbar_ax = fig.add_subplot(gs[:, 2])
cbar_diff = fig.colorbar(hm_top_right.collections[0], cax=cbar_ax, shrink=0.4, extend="max")
cbar_diff.set_label(r"Absolute probability difference [-]", size=fontsize)
# %%


fig = plt.figure(figsize=(11, 9), dpi=300)
gs = gridspec.GridSpec(2, 3, height_ratios=[1, 1], width_ratios=[1, 1, 0.05])
plt.subplots_adjust(wspace=0.4, hspace=0.3)

ax_heatmap_top_left = plt.subplot(gs[0, 0])
cg = sns.heatmap(
    co_occurrence_all,
    cmap=cmap,
    norm=norm_cmap,
    xticklabels=region_order,
    yticklabels=region_order,
    cbar=False,
)
cg.set_xticklabels(cg.get_xmajorticklabels(), fontsize=fontsize)
cg.set_yticklabels(cg.get_ymajorticklabels(), fontsize=fontsize, rotation=0)

cbar = plt.colorbar(cg.collections[0])
cbar.ax.tick_params(labelsize=16)
cbar.set_ticks(bounds)
cbar.set_label(r"P($E_{r,column}$|$E_{r,row}$)", size=fontsize)
plt.title("(a) All regimes", size=fontsize + 4, loc="left")

ax_heatmap_top_right = plt.subplot(gs[0, 1])
hm_top_right = plot_heatmap(ax_heatmap_top_right, "NAO -")
ax_heatmap_top_right.set_title("(b) NAO -", size=fontsize + 4, loc="left")

ax_heatmap_bottom_left = plt.subplot(gs[1, 0])
plot_heatmap(ax_heatmap_bottom_left, "Blocking")
ax_bar_bottom_left.set_title("(c) Blocking", size=fontsize + 4, loc="left")

ax_heatmap_bottom_right = plt.subplot(gs[1, 1])
plot_heatmap(ax_heatmap_bottom_right, "Atl. Ridge")
ax_bar_bottom_right.set_title("           (d) Atl. Ridge", size=fontsize + 4)

cbar_ax = fig.add_subplot(gs[:, 2])
cbar_diff = fig.colorbar(hm_top_right.collections[0], cax=cbar_ax, shrink=0.4, extend="max")
cbar_diff.set_label(r"Absolute probability difference [-]", size=fontsize)
