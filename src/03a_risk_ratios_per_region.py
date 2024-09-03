"""Calculate risk ratio's of energy droughts under different weather regimes for different regions.

Author: Bram van Duinen (bramvduinen@gmail.com)

"""

# %%
# IMPORTS
import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import utils as ut
from config import (
    CLUSTER_NAMES,
    ED_FILENAME_REGIONS,
    PATH_CLUSTERS,
    PATH_ED_REGIONS,
    REGIONS,
    VERSION,
)
from tqdm.notebook import tqdm

# %%
REGION_LIST = list(REGIONS.keys())
drop_no_regime = (
    True  # set to true if  you want to drop events that occur under 'No-regime' weather regime
)

date_now = datetime.now()
rundate = f"{date_now.year:04}{date_now.month:02}{date_now.day:02}"

df_clusters = pd.read_csv(
    os.path.join(PATH_CLUSTERS, VERSION, "df_clusters_full_ordered.csv"),
)
df_clusters["time"] = pd.to_datetime(df_clusters["time"])
df_clusters["time"] = df_clusters["time"].apply(
    lambda dt: dt.replace(hour=12, minute=0, second=0),
)  # set time to noon, to match df. Is daily average anyway

df_clusters["dominant_cluster"] = (
    df_clusters["Bayes_cluster"]
    .rolling(window=7, min_periods=7)
    .apply(ut.find_dominant_wr_v2, raw=False)
)

real_wrs = df_clusters["dominant_cluster"] <= 3  # ignore no-regime
cluster_occurrence_rate = df_clusters.loc[real_wrs, "dominant_cluster"].value_counts(
    normalize=True,
)

ed = pd.read_csv(
    os.path.join(PATH_ED_REGIONS, ED_FILENAME_REGIONS),
).reset_index(drop=True)
ed["run"] = ed["runs"].str.extract(r"(\d+)").astype(int)
df_events = ed.drop(["Unnamed: 0", "runs"], axis=1)


# %% define functions
def cluster_risk(df, region):
    """Calculate the risk ratio (w.r.t. climatology) of energy drought events within each weather regime for a given list of countries"""
    # disregarding events that occur under 'No-regime' weather regime!
    events_all = df.query("country == @region and 0 <= dominant_weather_regime <= 3")
    risks = []

    for i in range(4):
        events_wr_i = df.query("country == @region and dominant_weather_regime == @i")
        # n_events_wr_i / events_all gives the probability of an event under weather regime i
        # cluster_occurrence_rate[i] gives the probability of having weather regime i
        r_ci = (len(events_wr_i) / len(events_all)) / (cluster_occurrence_rate[i])
        risks.append(r_ci)

    return np.array(risks)


def monte_carlo_risk(df_event_wr, n_sample=100, n_iter=100):
    """Calculate the mean and standard deviation of the risk ratios for a given number of Monte Carlo iterations."""
    risk_ratios_all = np.empty((n_iter, 6, 4))
    for i in tqdm(range(n_iter)):
        random_runs = np.random.choice(np.arange(10, 170), size=n_sample, replace=False)
        selection_counts = df_event_wr.query("run in @random_runs")
        risk_ratios = []
        for region in REGION_LIST:
            risk_ratios.append(cluster_risk(selection_counts, region) - 1)
        risk_ratios_all[i] = risk_ratios
    mean_risk_ratios = np.mean(risk_ratios_all, axis=0)
    std_risk_ratios = np.std(risk_ratios_all, axis=0)
    return mean_risk_ratios, std_risk_ratios


def plot_risk_ratio(risk_ratios, std_risk, ax):
    """Plot the risk ratios for each region and weather regime."""
    labels = [CLUSTER_NAMES[0], CLUSTER_NAMES[1], CLUSTER_NAMES[2], CLUSTER_NAMES[3]]
    num_conditions = len(labels)
    width = 0.15
    positions = np.arange(num_conditions)

    region_colors = {
        "Nordic": "C0",
        "Baltic": "cyan",
        "Northwestern": "C1",
        "Central": "C2",
        "Eastern": "C3",
        "Iberia": "C4",
    }
    # plt.figure(dpi=300, figsize=(8, 6))
    ax.bar(
        positions - 3 * width,
        risk_ratios[2],
        width,
        yerr=std_risk[2],
        capsize=6.5,
        label="Northern",
        color=region_colors["Nordic"],
        edgecolor="black",
        bottom=1,
    )
    ax.bar(
        positions - 2 * width,
        risk_ratios[3],
        width,
        yerr=std_risk[3],
        capsize=6.5,
        label="Baltic",
        color=region_colors["Baltic"],
        edgecolor="black",
        bottom=1,
    )
    ax.bar(
        positions - width,
        risk_ratios[1],
        width,
        yerr=std_risk[1],
        capsize=6.5,
        label="Northwestern",
        color=region_colors["Northwestern"],
        edgecolor="black",
        bottom=1,
    )
    ax.bar(
        positions,
        risk_ratios[4],
        width,
        yerr=std_risk[4],
        capsize=6.5,
        label="Central",
        color=region_colors["Central"],
        edgecolor="black",
        bottom=1,
    )
    ax.bar(
        positions + width,
        risk_ratios[5],
        width,
        yerr=std_risk[5],
        capsize=6.5,
        label="Eastern",
        color=region_colors["Eastern"],
        edgecolor="black",
        bottom=1,
    )
    ax.bar(
        positions + 2 * width,
        risk_ratios[0],
        width,
        yerr=std_risk[0],
        capsize=6.5,
        label="Iberia",
        color=region_colors["Iberia"],
        edgecolor="black",
        bottom=1,
    )

    ax.axhline(y=1, color="k", linestyle="-", linewidth=0.8)

    ax.set_ylabel("Risk ratio [-]", fontsize=14)

    ax.set_yscale("log")
    ax.set_yticks([0.2, 0.5, 1, 2, 5])
    ax.set_yticklabels([0.2, 0.5, 1, 2, 5], fontsize=14)
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, fontsize=14)
    ax.legend(fontsize=12)

    for pos in positions[1:]:  # Skip the first position
        ax.axvline(x=pos - 3.75 * width, color="k", linestyle="--", linewidth=0.5)


# %% first for all 1600 events
df_event_wr = ut.find_dominant_wr(df_events, df_clusters, cluster_col="Bayes_cluster")

risk_ratios, std_risk = monte_carlo_risk(df_event_wr)

fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
plot_risk_ratio(risk_ratios, std_risk, ax)

# %% then check sensitivity to the intensity of events, with higher return periods
num_events = [800, 320, 160, 64]
return_period = [2, 5, 10, 25]  # years
alphabet = "abcd"
risk_ratio_dict = {}

fig, axs = plt.subplots(2, 2, figsize=(15, 10), dpi=300)
axs = axs.flatten()
for i, n in enumerate(num_events):
    event_selection = df_events[df_events["event_number"] <= n]
    df_event_wr_selection = ut.find_dominant_wr(
        event_selection,
        df_clusters,
        cluster_col="Bayes_cluster",
    )

    risk_ratios, std_risk = monte_carlo_risk(df_event_wr_selection)
    risk_ratios[risk_ratios == -1] = (
        np.nan
    )  # set to nan if no events are present under this weather regime
    plot_risk_ratio(risk_ratios, std_risk, axs[i])
    axs[i].set_title(
        f"({alphabet[i]}) Return period: {return_period[i]} years",
        loc="left",
        fontsize=16,
    )
    axs[i].set_yticks([0.01, 0.2, 0.5, 1, 2, 5])
    axs[i].set_yticklabels([0.01, 0.2, 0.5, 1, 2, 5], fontsize=14)
    risk_ratio_dict[return_period[i]] = risk_ratios

plt.tight_layout()
plt.show()
# %%
