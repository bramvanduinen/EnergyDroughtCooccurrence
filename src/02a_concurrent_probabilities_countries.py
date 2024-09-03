"""Calculate the co-occurrence probabilities of energy drought events for all possible country combinations.
Plot clustermap of countries, with the co-occurrence probabilities as the values in the heatmap.
Save resulting row-order for further use.
Country clusters are used as regions in further analysis.

Author: Bram van Duinen (bramvduinen@gmail.com)

"""

# %%
# IMPORTS
import os
import pickle
from itertools import product

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import utils as ut
from config import (
    COUNTRIES,
    ED_FILENAME_COUNTRIES,
    PATH_CLUSTERS,
    PATH_DATA,
    PATH_ED,
    VERSION,
)
from matplotlib import colors
from tqdm.notebook import tqdm

# %%
# LOAD DATA
HOMEDIR = "/usr/people/duinen/MSc-thesis/"
ROW_ORDER = np.load(
    f"{HOMEDIR}Data/row_order_nettodemand_v20240220.npy",
)  # load the row ordering of the clustered residual heatmap, to follow the same clustering!
# if not yet calculated, use row_order calculated in this script.

# %%
ut.check_make_dir(os.path.join(PATH_DATA, VERSION))
DF_CO_OCCURRENCES_FILENAME = os.path.join(PATH_DATA, VERSION, "df_co_occurrences.csv")
COUNTS_PER_RUN_FILENAME = os.path.join(PATH_DATA, VERSION, "counts_per_run.csv")

df_clusters = pd.read_csv(
    os.path.join(PATH_CLUSTERS, VERSION, "df_clusters_full_ordered.csv"),
)
df_clusters["time"] = pd.to_datetime(df_clusters["time"])
df_clusters["time"] = df_clusters["time"].apply(
    lambda dt: dt.replace(hour=12, minute=0, second=0),
)  # set time to noon, to match df. Is daily average anyway

ed = pd.read_csv(
    os.path.join(PATH_ED, ED_FILENAME_COUNTRIES),
).reset_index(drop=True)
ed["run"] = ed["runs"].str.extract(r"(\d+)").astype(int)
df_events = ed.drop(["Unnamed: 0", "runs"], axis=1)

# TODO: eventually, remove this line if the rest is not included in simulation!
df_events.query("country.isin(@COUNTRIES)", inplace=True)

# TODO: this find_dominant_wr can be replaced by functionality below, just sample at end_date
df_event_wr = ut.find_dominant_wr(df_events, df_clusters, cluster_col="Bayes_cluster")


def find_dominant_cluster(window):
    counts = window.value_counts()
    max_count = counts.max()
    # Check if the maximum count is at least 4
    if max_count >= 4:
        return counts.idxmax()
    # Return 5 if no cluster is dominant
    return 5


df_clusters["dominant_cluster"] = (
    df_clusters["Bayes_cluster"]
    .rolling(window=7, min_periods=7)
    .apply(find_dominant_cluster, raw=False)
)
cluster_occurrences = plt.hist(df_clusters["dominant_cluster"], bins=np.arange(7), density=True)[0]

# %%
# DEFINE FUNCTIONS


def get_co_occurrence(df_full, wr):
    """Find co-occurrences between country pairs for each run.
    Save dataframe of co-occurrence counts per country pair, per run.
    Later converted into co-occurrence probabilities.
    """
    overlapping_events_list = []
    co_occur_count_list = []

    if wr != "all":
        df = df_full.query("dominant_weather_regime == @wr")
        num_events_for_wr = (
            df_events.query("dominant_weather_regime == @wr").groupby("country").size()
        )
    if wr == "all":
        num_events_for_wr = df_events.groupby("country").size()
        df = df_full.copy()

    grouped_runs = df.groupby("run")

    for run, run_group in tqdm(grouped_runs):
        country_combinations = product(df_full["country"].unique(), repeat=2)
        for country_1, country_2 in country_combinations:
            co_occur = 0  # reset counter
            events_country_1 = run_group[run_group["country"] == country_1].sort_values(
                by="start_time",
            )
            events_country_2 = df_full.query("run == @run & country == @country_2").sort_values(
                by="start_time",
            )

            for _, event_country_1 in events_country_1.iterrows():
                for _, event_country_2 in events_country_2.iterrows():
                    if (
                        event_country_1["end_time"] >= event_country_2["start_time"]
                        and event_country_2["end_time"] >= event_country_1["start_time"]
                    ):
                        co_occur += 1

                        overlapping_events_list.append(
                            {
                                "country_1": country_1,
                                "event_number_1": event_country_1["event_number"],
                                "start_time_1": event_country_1["start_time"],
                                "end_time_1": event_country_1["end_time"],
                                "country_2": country_2,
                                "event_number_2": event_country_2["event_number"],
                                "start_time_2": event_country_2["start_time"],
                                "end_time_2": event_country_2["end_time"],
                                "run": run,
                            },
                        )

            co_occur_count_list.append(
                {
                    "country_1": country_1,
                    "country_2": country_2,
                    "run": run,
                    "count": co_occur,
                },
            )

    overlapping_events_df = pd.DataFrame(overlapping_events_list)
    co_occur_count_df_raw = pd.DataFrame(co_occur_count_list)

    co_occur_count_df = co_occur_count_df_raw.groupby(["country_1", "country_2"], as_index=False)[
        "count"
    ].sum()

    return overlapping_events_df, co_occur_count_df_raw, co_occur_count_df


def count_to_conditional(co_occur_count_df, country_order):
    """Convert co-occurrence counts to conditional probabilities."""
    pivot_table = (
        co_occur_count_df.pivot(index="country_1", columns="country_2", values="count")
        .reindex(country_order)
        .reindex(columns=country_order)
    )

    base_rates = np.diag(pivot_table).copy()

    conditional = pivot_table / base_rates[:, np.newaxis]
    return conditional, base_rates


def monte_carlo(counts_per_run, n_sample, n_iter, vmax=None):
    """Perform Monte Carlo sampling to get confidence intervals on the conditional probabilities."""
    count_per_run = pd.DataFrame(counts_per_run)
    conditional_probs = np.empty((n_iter, len(COUNTRIES), len(COUNTRIES)))

    for i in tqdm(range(n_iter)):
        random_runs = np.random.choice(np.arange(10, 170), size=n_sample, replace=False)
        selection_counts = count_per_run.query("run in @random_runs")
        co_occur_count_df = selection_counts.groupby(["country_1", "country_2"], as_index=False)[
            "count"
        ].sum()
        conditional, base_rates = count_to_conditional(
            co_occur_count_df,
            np.array(COUNTRIES)[ROW_ORDER],
        )
        conditional_probs[i] = conditional

    meanprob = np.mean(conditional_probs, axis=0)
    std = np.std(conditional_probs, axis=0)

    cg = sns.heatmap(
        (2 * std) / meanprob,
        cmap="Reds",
        xticklabels=np.array(COUNTRIES)[ROW_ORDER],
        yticklabels=np.array(COUNTRIES)[ROW_ORDER],
        vmax=vmax,
    )
    cbar = cg.collections[0].colorbar
    cbar.set_label("2*sigma/mean")
    return meanprob, std


# %% CO-OCCURRENCE ANALYSIS
if not os.path.exists(DF_CO_OCCURRENCES_FILENAME):
    print("Computing co-occurrences...")
    events_df, counts_per_run, co_occur_counts_df = get_co_occurrence(df_full=df_events, wr="all")
    co_occur_counts_df.to_csv(DF_CO_OCCURRENCES_FILENAME, index=False)
    pd.DataFrame(counts_per_run).to_csv(COUNTS_PER_RUN_FILENAME)
else:
    print("Loading previously computed co-occurrences...")
    co_occur_counts_df = pd.read_csv(DF_CO_OCCURRENCES_FILENAME)
    counts_per_run = pd.read_csv(COUNTS_PER_RUN_FILENAME)

# %% MONTE CARLO SAMPLING OF CO-OCCURRENCE, FOR CONFIDENCE
meanprob, std = monte_carlo(counts_per_run, n_sample=100, n_iter=100)

# %%
# PLOT HEATMAP ALL WEATHER REGIMES
co_occur_counts_df["country_1_index"] = co_occur_counts_df["country_1"]
co_occur_counts_df["country_2_index"] = co_occur_counts_df["country_2"]

co_occur_counts_df["prob"] = pd.to_numeric(co_occur_counts_df["count"]) / 1600

heatmap_data = co_occur_counts_df.pivot(
    index=["country_1_index"],
    columns=["country_2_index"],
    values="prob",
)

bounds = np.arange(0, 1.1, 0.1)

# Create a colormap with the specified boundaries
cmap = colors.ListedColormap(sns.color_palette("Reds", len(bounds) - 1))
norm_cmap = colors.BoundaryNorm(bounds, cmap.N)

plt.figure(figsize=(11, 8), dpi=300)
cg = sns.clustermap(
    heatmap_data.values,
    cmap=cmap,
    norm=norm_cmap,
    xticklabels=COUNTRIES,
    yticklabels=COUNTRIES,
)
cg.ax_heatmap.set_xticklabels(cg.ax_heatmap.get_xmajorticklabels(), fontsize=16)
cg.ax_heatmap.set_yticklabels(cg.ax_heatmap.get_ymajorticklabels(), fontsize=16)


cg.ax_col_dendrogram.set_visible(False)
x0, _y0, _w, _h = cg.cbar_pos
cg.ax_cbar.set_position([1.02, 0.06, 0.025, 0.74])
cg.cax.set_ylabel(r"P(X|Y)", size=18, rotation=90, labelpad=20)
cg.cax.yaxis.set_label_coords(x=3.5, y=0.5)
cg.cax.tick_params(labelsize=14)
cg.cax.yaxis.set_ticks(bounds)
row_order = cg.dendrogram_row.reordered_ind

for a in cg.ax_row_dendrogram.collections:
    a.set_linewidth(2)

# %% Check sensitivity to return period, i.e. with stronger events

num_events = [800, 320, 160, 64]
return_period = [2, 5, 10, 25]  # years
co_occurrence_dict = {}

for i, n in enumerate(num_events):
    event_selection = df_events[df_events["event_number"] <= n]
    events_df, counts_per_run, co_occur_counts_df = get_co_occurrence(
        df_full=event_selection,
        wr="all",
    )
    meanprob, std = monte_carlo(counts_per_run, n_sample=100, n_iter=100)

    co_occurrence_dict[return_period[i]] = {
        "meanprob": meanprob,
        "std": std,
    }

    plt.figure(figsize=(11, 8), dpi=300)
    plt.title(f"Return period: {return_period[i]} years", size=20)
    cg = sns.heatmap(
        meanprob,
        cmap=cmap,
        norm=norm_cmap,
        xticklabels=np.array(COUNTRIES)[ROW_ORDER],
        yticklabels=np.array(COUNTRIES)[ROW_ORDER],
        cbar=False,
    )
    cg.set_xticklabels(cg.get_xmajorticklabels(), fontsize=16, rotation=90)
    cg.set_yticklabels(cg.get_ymajorticklabels(), fontsize=16)
    cbar = plt.colorbar(cg.collections[0])
    cbar.set_label(r"P(X|Y)", size=18)
    plt.show()

FILENAME_CO_OCCURRENCE_SENSITIVITY = os.path.join(
    PATH_DATA,
    VERSION,
    "co_occurrence_sensitive.pickle",
)
with open(FILENAME_CO_OCCURRENCE_SENSITIVITY, "wb") as f:
    pickle.dump(co_occurrence_dict, f)
