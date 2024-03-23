# %%
# IMPORTS
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import utils as ut
from config import CLUSTER_NAMES, PATH_CLUSTERS, PATH_ED, REGIONS, VERSION

# %%
# LOAD DATA
df_clusters = pd.read_csv(
    os.path.join(PATH_CLUSTERS, VERSION, "df_clusters_full_ordered.csv"),
)
df_clusters["time"] = pd.to_datetime(df_clusters["time"])
df_clusters["time"] = df_clusters["time"].apply(
    lambda dt: dt.replace(hour=12, minute=0, second=0),
)  # set time to noon, to match df. Is daily average anyway

ed = pd.read_csv(
    os.path.join(PATH_ED, "netto_demand_el7_winter_LENTIS_2023_PD_1600_events.csv"),
).reset_index(drop=True)
ed["run"] = ed["runs"].str.extract(r"(\d+)").astype(int)
df_events = ed.drop(["Unnamed: 0", "runs"], axis=1)

random_ed = pd.read_csv(
    os.path.join(PATH_ED, "random_netto_demand_el7_winter_LENTIS_2023_PD_1600_events.csv"),
).reset_index(drop=True)
random_ed["run"] = random_ed["runs"].str.extract(r"(\d+)").astype(int)
df_random = random_ed.drop(["Unnamed: 0", "runs"], axis=1)

df_event_wr = ut.find_dominant_wr(df_events, df_clusters, cluster_col="Bayes_cluster")
df_random_wr = ut.find_dominant_wr(df_random, df_clusters, cluster_col="Bayes_cluster")

# %%


def compare_wr_strength(df_event_wr, df_random_wr, region_key, plottype="kde"):
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
for region_key in REGIONS:
    region = REGIONS[region_key]
    df_event_region = df_event_wr.query("country in @region")
    df_random_region = df_random_wr.query("country in @region")

    compare_wr_strength(df_event_region, df_random_region, region_key, plottype="kde")

# %%
max_wri_events = np.array(df_event_wr["max_wri_at_dominant_regime"], dtype=float)
plt.hist(max_wri_events)

# %%
