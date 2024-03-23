# %%
# IMPORTS
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import utils as ut
from config import CLUSTER_NAMES, PATH_CLUSTERS, PATH_ED, VERSION
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


def persistence_comparison(cluster_id, df_event_wr, df_random_wr, df, streak_lengths_Bayes):
    """Calculate and plot the mean streak lengths for a given weather regime cluster,
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

    df_events_regime = df_event_wr.query("dominant_weather_regime == @cluster_id")
    df_random_regime = df_random_wr.query("dominant_weather_regime == @cluster_id")
    df_list = [df_events_regime, df_random_regime]
    full_mean = []
    full_median = []
    err = [0, 0]
    for i, df_i in enumerate(df_list):
        for ind, row in tqdm(df_i.iterrows(), total=df_i.shape[0]):
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
            #     streak_lengths_copy = streak_lengths_copy.drop(streak_length.index)
            except IndexError:
                err[i] += 1
                continue
        full_mean.append(np.mean(streak_length_array[i]))
        full_median.append(np.median(streak_length_array[i]))

    diff = 100 * (full_mean[0] - full_mean[1]) / full_mean[1]
    diff_median = 100 * (full_median[0] - full_median[1]) / full_median[1]

    plt.figure()
    bins = np.arange(49, step=3)
    n_random_events = len(df_random_regime)
    n_events = len(df_events_regime)
    if n_random_events > n_events:
        plt.hist(streak_length_array[1], bins=bins, color="C0", alpha=0.5)
        plt.hist(streak_length_array[0], bins=bins, color="r", alpha=0.5)
    elif n_random_events <= n_events:
        plt.hist(streak_length_array[0], bins=bins, color="r", alpha=0.5)
        plt.hist(streak_length_array[1], bins=bins, color="C0", alpha=0.5)
    plt.axvline(
        full_mean[0],
        color="r",
        linestyle="dashed",
        linewidth=1,
        label=f"Event mean (n = {n_events})",
    )
    plt.axvline(
        full_mean[1],
        color="C0",
        linestyle="dashed",
        linewidth=1,
        label=f"Full regime mean (n = {n_random_events})",
    )
    plt.axvline(
        full_median[0],
        color="r",
        linestyle="dotted",
        linewidth=1,
        label=f"Event median (n = {n_events})",
    )
    plt.axvline(
        full_median[1],
        color="C0",
        linestyle="dotted",
        linewidth=1,
        label=f"Full regime median (n = {n_random_events})",
    )
    plt.xlabel("Persistence [days]")
    plt.ylabel("Count")
    plt.title(
        f"Regime: {CLUSTER_NAMES[cluster_id]}; Difference: {diff:.2f}% (mean) or {diff_median:.2f}% (median)",
    )
    plt.legend()
    plt.show()

    return full_mean, err


# %%
# ANALYSIS

pers_stats, streak_lengths = calculate_persistency_stats(
    df_clusters,
    cluster_col="Bayes_cluster",
)

df_event_wr = ut.find_dominant_wr(df_events, df_clusters, cluster_col="Bayes_cluster")
df_random_wr = ut.find_dominant_wr(df_random, df_clusters, cluster_col="Bayes_cluster")

# %%
# TODO: Turn this into a method, repeat analysis for regions like in 02b.
event_mean_list = []
full_mean_list = []
for i in range(4):
    # for i in [1, 2]:
    event_mean, full_mean = persistence_comparison(
        i,
        df_event_wr,
        df_random_wr,
        df_clusters,
        streak_lengths,
    )
    event_mean_list.append(event_mean)
    full_mean_list.append(full_mean)

# %%
