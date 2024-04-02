# %%
# IMPORTS
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import utils as ut
from config import CLUSTER_NAMES, PATH_CLUSTERS, PATH_ED, REGIONS, VERSION
from tqdm import tqdm

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
# ed = pd.read_csv(
#     os.path.join(PATH_ED, "random_v2_netto_demand_el7_winter_LENTIS_2023_PD_1600_events.csv"),
# ).reset_index(drop=True)
ed["run"] = ed["runs"].str.extract(r"(\d+)").astype(int)
df_events = ed.drop(["Unnamed: 0", "runs"], axis=1)

random_ed = pd.read_csv(
    os.path.join(PATH_ED, "random_netto_demand_el7_winter_LENTIS_2023_PD_1600_events.csv"),
).reset_index(drop=True)
random_ed["run"] = random_ed["runs"].str.extract(r"(\d+)").astype(int)
df_random = random_ed.drop(["Unnamed: 0", "runs"], axis=1)


# %%
# Defining functions
def calculate_persistency_stats(df, date_col="time", cluster_col="cluster_id"):
    df[date_col] = pd.to_datetime(df[date_col])

    # Calculate the difference in days within each cluster
    df["date_diff"] = df.groupby(cluster_col)[date_col].diff().dt.days
    df["new_streak"] = df["date_diff"] != 1

    # Create a new group identifier for each streak
    df["streak_id"] = df.groupby(cluster_col)["new_streak"].cumsum()

    # Calculate the length of each streak
    streak_lengths = df.groupby([cluster_col, "streak_id"]).size().reset_index(name="streak_length")

    # Calculate the average, median, standard deviation, and maximum persistency
    persistency_stats = streak_lengths.groupby(cluster_col).agg(["mean", "median", "std", "max"])

    return persistency_stats, streak_lengths


def calc_persistence_comparison(
    cluster_id,
    region,
    df_event_wr,
    df_random_wr,
    df,
    streak_lengths_Bayes,
):
    """Calculate the mean streak lengths for a given weather regime cluster,
    comparing event-specific means to the overall mean streak length for that cluster.

    Args:
    ----
    cluster_id (int): The ID of the cluster whose streak lengths are to be analyzed.
    df_event_wr (DataFrame): DataFrame containing event and weather regime information.
    df (DataFrame): DataFrame containing just weather regime data for matching and querying purposes.
    streak_lengths_Bayes (DataFrame): DataFrame containing streak lengths for Bayesian clusters.

    Returns:
    -------
    tuple: A tuple containing the mean streak length for the events and the overall mean streak length.

    """
    streak_lengths_copy = streak_lengths_Bayes.copy()

    streak_length_array = ([], [])

    df_events_regime = df_event_wr.query(
        "country in @region and dominant_weather_regime == @cluster_id",
    )
    df_random_regime = df_random_wr.query(
        "country in @region and dominant_weather_regime == @cluster_id",
    )
    df_list = [df_events_regime, df_random_regime]
    err = [0, 0]
    for i, df_i in enumerate(df_list):
        for ind, row in df_i.iterrows():
            run = row["run"]
            start_time = pd.to_datetime(row["start_time"])

            first_time_index = np.where(np.array(row["weather_regime_ids"]) == cluster_id)[0][0]
            streak_time = start_time + pd.Timedelta(days=first_time_index)

            try:
                streak_id = df.query("run == @run and time == @streak_time")["streak_id"].values[0]
                streak_length = streak_lengths_copy.query(
                    "Bayes_cluster == @cluster_id and streak_id == @streak_id",
                )
                streak_length_value = streak_length["streak_length"].values[0]
                streak_length_array[i].append(streak_length_value)
            except IndexError:
                err[i] += 1
                continue
    return streak_length_array


def plot_all_persistence_comparison(df_event_wr, df_random_wr, df, streak_lengths_Bayes):
    for region_key in REGIONS:
        region = REGIONS[region_key]

        fig, axs = plt.subplots(2, 2)
        axs = axs.flatten()
        bins = np.arange(49, step=3)

        for i in tqdm(range(4)):
            streak_length_array = calc_persistence_comparison(
                i,
                region,
                df_event_wr,
                df_random_wr,
                df,
                streak_lengths_Bayes,
            )

            diff = (
                100
                * (np.mean(streak_length_array[0]) - np.mean(streak_length_array[1]))
                / np.mean(
                    streak_length_array[1],
                )
            )

            axs[i].hist(streak_length_array[1], bins=bins, color="C0", alpha=0.5, label="Random")
            axs[i].hist(streak_length_array[0], bins=bins, color="r", alpha=0.5, label="Events")
            axs[i].axvline(
                np.mean(streak_length_array[0]),
                color="r",
                linestyle="dashed",
                linewidth=1,
            )
            axs[i].axvline(
                np.mean(streak_length_array[1]),
                color="C0",
                linestyle="dashed",
                linewidth=1,
            )
            axs[i].set_xlabel("Persistence [days]")
            axs[i].set_ylabel("Count")
            axs[i].set_xticks(np.arange(0, 50, step=5))
            axs[i].set_title(
                f"{CLUSTER_NAMES[i]}, diff = {diff:.2f}%",
            )
            axs[i].legend(loc="upper right")

        plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust the layout so that the plots do not overlap
        plt.suptitle(f"Region: {region_key}")
        plt.show()


# %%
# ANALYSIS

pers_stats, streak_lengths = calculate_persistency_stats(
    df_clusters,
    cluster_col="Bayes_cluster",
)

df_event_wr = ut.find_dominant_wr(df_events, df_clusters, cluster_col="Bayes_cluster")
df_random_wr = ut.find_dominant_wr(df_random, df_clusters, cluster_col="Bayes_cluster")


# %%
plot_all_persistence_comparison(df_event_wr, df_random_wr, df_clusters, streak_lengths)
