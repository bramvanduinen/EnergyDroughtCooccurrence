# %%
# IMPORTS
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import utils as ut
from config import CLUSTER_NAMES, PATH_CLUSTERS, PATH_ED, VERSION
from tqdm import tqdm

# %%
# LOAD DATA
df = pd.read_csv(os.path.join(PATH_CLUSTERS, VERSION, "df_clusters_full_ordered.csv"), index_col=0)
df.drop(columns=["Unnamed: 0"], inplace=True)
df["time"] = pd.to_datetime(df["time"])
df["time"] = df["time"].apply(
    lambda dt: dt.replace(hour=12, minute=0, second=0),
)  # set time to noon, to match df. Is daily average anyway

ed = pd.read_csv(
    os.path.join(PATH_ED, "netto_demand_el7_winter_LENTIS_2023_PD_1600_events.csv"),
).reset_index(drop=True)
ed["run"] = ed["runs"].str.extract(r"(\d+)").astype(int)
df_events = ed.drop(["Unnamed: 0", "runs"], axis=1)


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


def persistence_comparison(cluster_id, df_event_wr, df, streak_lengths_Bayes):
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

    streak_length_array = []

    df_events_regime = df_event_wr.query("dominant_weather_regime == @cluster_id")

    for ind, row in tqdm(df_events_regime.iterrows(), total=df_events_regime.shape[0]):
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
            streak_length_array.append(streak_length_value)
            streak_lengths_copy = streak_lengths_copy.drop(streak_length.index)
        except IndexError:
            continue

    streak_length_all = streak_lengths_Bayes.query("Bayes_cluster == @cluster_id")[
        "streak_length"
    ].values
    full_mean = np.mean(streak_length_all)
    event_mean = np.mean(streak_length_array)

    plt.figure()
    plt.hist(streak_length_all, color="C0", alpha=0.3)
    plt.hist(streak_length_array, color="r", alpha=0.5)
    plt.axvline(event_mean, color="r", linestyle="dashed", linewidth=1, label="Event mean")
    plt.axvline(full_mean, color="C0", linestyle="dashed", linewidth=1, label="Full regime mean")
    plt.xlabel("Persistence [days]")
    plt.ylabel("Count")
    plt.title(f"Regime: {CLUSTER_NAMES[cluster_id]}")
    plt.legend()
    plt.show()

    return event_mean, full_mean


# %%
# ANALYSIS

pers_stats, streak_lengths = calculate_persistency_stats(
    df,
    cluster_col="Bayes_cluster",
)

df_event_wr = ut.find_dominant_wr(df_events, df, cluster_col="Bayes_cluster")

event_mean_list = []
full_mean_list = []
for i in range(4):
    event_mean, full_mean = persistence_comparison(i, df_event_wr, df, streak_lengths)
    event_mean_list.append(event_mean)
    full_mean_list.append(full_mean)

# %%
# GENERAL STATISTICS
colors = ["C0", "C1", "C2", "C3", "C4"]

for cluster_id in range(5):
    color = colors[cluster_id]
    cluster_data_Bayes = streak_lengths[streak_lengths["Bayes_cluster"] == cluster_id][
        "streak_length"
    ]
    plt.axvline(np.mean(cluster_data_Bayes), color=color)
    cluster_data = streak_lengths[streak_lengths["cluster_id"] == cluster_id]["streak_length"]
    plt.axvline(np.mean(cluster_data), color=color, linestyle="dotted")
    sns.kdeplot(
        cluster_data_Bayes,
        color=color,
        label=f"Cluster {cluster_id}",
        bw_method=0.5,
        clip=(0, 100),
    )
    sns.kdeplot(cluster_data, color=color, linestyle="dotted", bw_method=0.5, clip=(0, 100))

# Add labels and title
plt.xlabel("Streak Length [days]")
plt.ylabel("Density")
plt.title("KDE Plot of Streak Lengths by Cluster (dotted = default, solid = Bayes)")
plt.xlim(0, 20)
plt.legend()

# %%
# Set up the figure and axis
plt.figure(figsize=(10, 6))
ax = plt.gca()

# Plot histograms for each cluster with different colors
for cluster_id in range(5):
    cluster_data2 = streak_lengths[streak_lengths["Bayes_cluster"] == cluster_id]["streak_length"]
    plt.hist(
        cluster_data2,
        bins=np.arange(30),
        alpha=0.7,
        label=f"Cluster {cluster_id}",
        density=True,
    )

# Add labels and title
plt.xlabel("Streak Length [days]")
plt.ylabel("Frequency")
plt.title("Histogram of Streak Lengths by Cluster")
plt.legend()

# %%
for i in range(5):
    print(
        df.query("cluster_id == @i")["correlation"].mean(),
        df.query("cluster_id == @i")["correlation"].std(),
    )
    print(
        df.query("Bayes_cluster == @i")["correlation"].mean(),
        df.query("Bayes_cluster == @i")["correlation"].std(),
    )

# %%
for i in range(5):
    sns.kdeplot(df.query("Bayes_cluster == @i")["correlation"], label=f"Cluster {i}")
plt.legend()


# %%
# APPENDIX
# Not used right now, but it's an option
def cluster_length_per_country(cluster_id, country):
    streak_length_array = []
    df_events_regime = df_event_wr.query(
        "country == @country and dominant_weather_regime == @cluster_id",
    )
    for ind, row in tqdm(df_events_regime.iterrows()):
        run = row["run"]
        start_time = pd.to_datetime(row["start_time"])
        first_time = np.where(np.array(row["weather_regime_ids"]) == cluster_id)[0][0]
        streak_time = start_time + pd.Timedelta(days=first_time)
        try:
            streak_id = df.query("run == @run and time == @streak_time")["streak_id"].values[0]
            streak_length = streak_lengths.query(
                "Bayes_cluster == @cluster_id and streak_id == @streak_id",
            )
            streak_length_array.append(streak_length["streak_length"].values[0])
        except IndexError:
            continue

    plt.figure()
    streak_length_all = streak_lengths.query("Bayes_cluster == @cluster_id")["streak_length"].values
    full_mean = np.mean(streak_length_all)
    event_mean = np.mean(streak_length_array)
    plt.hist(streak_length_all, color="C0", alpha=0.3)
    plt.hist(streak_length_array, color="r", alpha=0.5)
    plt.axvline(event_mean, color="r", linestyle="dashed", linewidth=1, label="Event mean")
    plt.axvline(full_mean, color="C0", linestyle="dashed", linewidth=1, label="Full regime mean")
    plt.xlabel("Persistence [days]")
    plt.ylabel("Count")
    plt.title(f"Regime: {CLUSTER_NAMES[cluster_id]}")
    plt.legend()
    plt.show()

    return event_mean, full_mean
