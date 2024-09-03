# %%
# IMPORTS
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import utils as ut
from config import (
    CLUSTER_NAMES,
    ED_FILENAME_REGIONS,
    ED_FILENAME_REGIONS_RANDOM,
    PATH_CLUSTERS,
    PATH_ED_REGIONS,
    REGIONS,
    VERSION,
)
from matplotlib.patches import Patch
from scipy import stats

# %%
# LOAD DATA
df_clusters = pd.read_csv(
    os.path.join(PATH_CLUSTERS, VERSION, "df_clusters_full_ordered.csv"),
)
df_clusters["time"] = pd.to_datetime(df_clusters["time"])
df_clusters["time"] = df_clusters["time"].apply(
    lambda dt: dt.replace(hour=12, minute=0, second=0),
)  # set time to noon, to match energy data; is daily average anyway

ed = pd.read_csv(
    os.path.join(PATH_ED_REGIONS, ED_FILENAME_REGIONS),
).reset_index(drop=True)

ed["run"] = ed["runs"].str.extract(r"(\d+)").astype(int)
df_events = ed.drop(["Unnamed: 0", "runs"], axis=1)

random_ed = pd.read_csv(
    os.path.join(PATH_ED_REGIONS, ED_FILENAME_REGIONS_RANDOM),
).reset_index(drop=True)
random_ed["run"] = random_ed["runs"].str.extract(r"(\d+)").astype(int)
df_random = random_ed.drop(["Unnamed: 0", "runs"], axis=1)

df_event_wr = ut.find_dominant_wr(df_events, df_clusters, cluster_col="Bayes_cluster")
df_random_wr = ut.find_dominant_wr(df_random, df_clusters, cluster_col="Bayes_cluster")


# %%
def compare_wr_strength(df_event_wr, df_random_wr, i):
    df_event_regime = df_event_wr.query("dominant_weather_regime == @i")
    df_random_regime = df_random_wr.query("dominant_weather_regime == @i")
    wri_events = np.array(df_event_regime["max_wri_at_dominant_regime"], dtype=float)
    wri_random = np.array(df_random_regime["max_wri_at_dominant_regime"], dtype=float)

    event_strength = np.nanmean(wri_events)
    random_strength = np.nanmean(wri_random)
    mean_diff = event_strength - random_strength
    perc_diff = 100 * mean_diff / random_strength
    t_stat, p_value = stats.ttest_ind(wri_events, wri_random, equal_var=False)

    n1, n2 = len(wri_events), len(wri_random)
    s1, s2 = np.var(wri_events, ddof=1), np.var(wri_random, ddof=1)

    # Calculate the standard error
    se = np.sqrt(s1 / n1 + s2 / n2)
    perc_se = 100 * se / random_strength
    ci_low = mean_diff - 1.96 * se  # 95% CI
    ci_high = mean_diff + 1.96 * se  # 95% CI
    perc_ci_low = perc_diff - 1.96 * perc_se  # 95% CI
    perc_ci_high = perc_diff + 1.96 * perc_se  # 95% CI

    return {
        "Mean Difference": mean_diff,
        "Percentage Difference": perc_diff,
        "P-Value": p_value,
        "95% CI (Low)": ci_low,
        "95% CI (High)": ci_high,
        "95% CI Percentage (Low)": perc_ci_low,
        "95% CI Percentage (High)": perc_ci_high,
    }


def compare_plot_wr_strength(df_event_wr, df_random_wr, region_key, plottype="kde"):
    bins = np.arange(-4, 4.5, 0.5)
    fig, axs = plt.subplots(2, 2)  # Create a 2x2 grid of subplots
    axs = axs.flatten()  # Flatten the array of axes for easier indexing

    for i in range(4):
        df_event_regime = df_event_wr.query("dominant_weather_regime == @i")
        df_random_regime = df_random_wr.query("dominant_weather_regime == @i")
        num_events = df_event_regime.shape[0]
        num_non_events = df_random_regime.shape[0]
        wri_events = np.array(df_event_regime["max_wri_at_dominant_regime"], dtype=float)
        wri_non_events = np.array(df_random_regime["max_wri_at_dominant_regime"], dtype=float)

        mean_wri_events = np.nanmean(wri_events)
        mean_wri_non_events = np.nanmean(wri_non_events)

        axs[i].axvline(0, color="k", linewidth=0.5)
        if num_events >= num_non_events:
            if plottype == "kde":
                sns.kdeplot(wri_events, ax=axs[i], color="C0", label=f"events (n = {num_events})")
                sns.kdeplot(
                    wri_non_events,
                    ax=axs[i],
                    color="C1",
                    label=f"random (n = {num_non_events})",
                )
            elif plottype == "hist":
                axs[i].hist(wri_events, bins=bins, color="C0", label=f"events (n = {num_events})")
                axs[i].hist(
                    wri_non_events,
                    bins=bins,
                    color="C1",
                    label=f"random (n = {num_non_events})",
                )
        elif num_events < num_non_events:
            if plottype == "kde":
                sns.kdeplot(
                    wri_non_events,
                    ax=axs[i],
                    color="C1",
                    label=f"random (n = {num_non_events})",
                )
                sns.kdeplot(wri_events, ax=axs[i], color="C0", label=f"events (n = {num_events})")
            elif plottype == "hist":
                axs[i].hist(
                    wri_non_events,
                    bins=bins,
                    color="C1",
                    label=f"random (n = {num_non_events})",
                )
                axs[i].hist(wri_events, bins=bins, color="C0", label=f"events (n = {num_events})")

        axs[i].axvline(mean_wri_events, color="C0", linestyle="dashed", linewidth=1)
        axs[i].axvline(mean_wri_non_events, color="C1", linestyle="dashed", linewidth=1)

        handles, labels = axs[i].get_legend_handles_labels()
        order = [
            labels.index(f"events (n = {num_events})"),
            labels.index(f"random (n = {num_non_events})"),
        ]
        axs[i].legend(
            [handles[idx] for idx in order],
            [labels[idx] for idx in order],
            loc="upper left",
        )
        axs[i].set_title(
            f"{CLUSTER_NAMES[i]}, diff = {100*(mean_wri_events - mean_wri_non_events)/mean_wri_non_events:.2f}%",
        )
        axs[i].set_xlabel(r"WRI $[\sigma]$")
        axs[i].set_ylabel("Density")
        if plottype == "kde":
            axs[i].set_ylim(0, 1)
        axs[i].set_xlim(-1, 4)

    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust the layout so that the plots do not overlap
    plt.suptitle(
        f"Comparison of WRI strengths for region: {region_key}",
    )  # Add a title to the figure
    plt.show()  # Display the figure


# %%
results_df = pd.DataFrame(
    columns=[
        "Region",
        "WR",
        "Mean Difference",
        "Percentage Difference",
        "P-Value",
        "95% CI (Low)",
        "95% CI (High)",
        "95% CI Percentage (Low)",
        "95% CI Percentage (High)",
    ],
)

result_rows = []

for region in REGIONS:
    df_event_region = df_event_wr.query("country in @region")
    df_random_region = df_random_wr.query("country in @region")
    for i in range(4):
        result = compare_wr_strength(df_event_region, df_random_region, i)

        row = {"Region": region, "WR": i, **result}
        result_rows.append(row)

results_df = pd.concat([results_df, pd.DataFrame(result_rows)], ignore_index=True)
# %%
max_wri_events = np.array(df_event_wr["max_wri_at_dominant_regime"], dtype=float)
plt.hist(max_wri_events)

# %%
colors = {
    "N": "C0",
    "B": "cyan",
    "NW": "C1",
    "C": "C2",
    "E": "C3",
    "IB": "C4",
}

regions = list(colors.keys())
region_labels = ["Northern", "Baltic", "Northwestern", "Central", "Eastern", "Iberia"]
weather_regimes = sorted(results_df["WR"].unique())

# first absolute values

fig, ax = plt.subplots(figsize=(10, 6))

bar_width = 0.8 / len(regions)

for i, condition in enumerate(weather_regimes):
    for j, region in enumerate(regions):
        # Filter data for this region and condition
        data = results_df[(results_df["WR"] == condition) & (results_df["Region"] == region)]

        mean_diff = data["Mean Difference"].values[0]
        ci_low = data["95% CI (Low)"].values[0]
        ci_high = data["95% CI (High)"].values[0]
        error = [
            [mean_diff - ci_low],
            [ci_high - mean_diff],
        ]

        # TODO: quick and dirty fix! Have to change it in the risk ratios calc.
        iplot = i
        if i == 0:
            iplot = 1
        elif i == 1:
            iplot = 0
        position = iplot - 0.4 + bar_width * (j + 0.5)

        bar = ax.bar(
            position,
            mean_diff,
            width=bar_width,
            color=colors[region],
            edgecolor="black",
            yerr=error,
            capsize=5,
            error_kw={"ecolor": "black", "capthick": 2},
        )

        # Apply hatching if not statistically significant
        p_value = data["P-Value"].values[0]
        if p_value > 0.05:
            bar[0].set_hatch("///")

ax.set_xticks(range(len(weather_regimes)))
ax.set_xticklabels(["NAO -", "NAO +", "Blocking", "Atl. Ridge"])
ax.set_ylabel(r"Difference in mean WRI $[\sigma]$")
ax.set_title("Difference in mean strength (events - random) with 95% CI")
ax.legend(region_labels, title="Region")
ax.axhline(0, color="black", linewidth=0.5)

plt.tight_layout()
plt.savefig("/usr/people/duinen/MSc-thesis/Results/Figures/strength_diff.png", dpi=300)
plt.show()

# %% now relative percentage change; use this one!

fig, axs = plt.subplots(1, 2, figsize=(18, 6), dpi=300)
bar_width = 0.8 / len(regions)

for k, ax in enumerate(axs):
    if k == 0:
        results_df_i = pd.read_csv("../Results/data/Bayes_full_v2/df_persistence_diff.csv")
        ax.set_ylabel(r"$\Delta$ Persistence [%]", size=18)
        ax.set_title("(a)", loc="left", size=18)
    elif k == 1:
        results_df_i = results_df
        ax.set_ylabel(r"$\Delta$ Strength [%]", size=18)
        ax.set_title("(b)", loc="left", size=18)
    for i, condition in enumerate(weather_regimes):
        for j, region in enumerate(regions):
            # Filter data for this region and condition
            data = results_df_i[
                (results_df_i["WR"] == condition) & (results_df_i["Region"] == region)
            ]
            mean_diff = data["Percentage Difference"].values[0]
            ci_low = data["95% CI Percentage (Low)"].values[0]
            ci_high = data["95% CI Percentage (High)"].values[0]
            error = [
                [mean_diff - ci_low],
                [ci_high - mean_diff],
            ]

            # TODO: quick and dirty fix! Have to change it in the risk ratios calc.
            iplot = i
            # if i == 0:
            #     iplot = 1
            # elif i == 1:
            #     iplot = 0
            position = iplot - 0.4 + bar_width * (j + 0.5)

            bar = ax.bar(
                position,
                mean_diff,
                width=bar_width,
                color=colors[region],
                edgecolor="black",
                yerr=error,
                capsize=5,
                error_kw={"ecolor": "black", "capthick": 2},
            )
        if i != 3:
            ax.axvline(x=position + 0.15, color="k", linestyle="--", linewidth=0.5)

            # Apply hatching if not statistically significant
            # p_value = data["P-Value"].values[0]
            # if p_value > 0.05:
            #     bar[0].set_hatch("///")

    ax.set_xticks(range(len(weather_regimes)))
    ax.set_xticklabels(["NAO +", "NAO \u2212", "Blocking", "Atl. Ridge"], size=18)
    ax.tick_params("y", labelsize=16)
    ax.set_ylim(-75, 40)
    # ax.set_title("Difference in mean strength (events - climatology) with 95% CI", size=18)
    legend_handles = [
        Patch(facecolor=colors[region], edgecolor="black", label=region_labels[i])
        for i, region in enumerate(regions)
    ]
    if k == 1:
        ax.legend(handles=legend_handles, fontsize=16)
    ax.axhline(0, color="black", linewidth=0.5)


# plt.tight_layout()

plt.savefig("/usr/people/duinen/MSc-thesis/Results/Figures/strength_diff_pct.png", dpi=300)
plt.show()

# %%
results_df.to_csv("../Results/data/Bayes_full_v2/df_strength_diff.csv")
