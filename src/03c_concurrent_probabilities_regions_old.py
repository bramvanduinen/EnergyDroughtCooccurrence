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
from config import CLUSTER_NAMES, COUNTRIES, PATH_CLUSTERS, PATH_DATA, REGIONS, VERSION
from matplotlib import cm, colors, gridspec
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.stats import norm
from tqdm.notebook import tqdm

# %%
# LOAD DATA
HOMEDIR = "/usr/people/duinen/MSc-thesis/"
ROW_ORDER = np.load(
    f"{HOMEDIR}Data/row_order_nettodemand_v20240220.npy",
)  # load the row ordering of the clustered residual heatmap, to follow the same clustering!
CMAP = "RdBu_r"

BOUNDS_GREY = np.arange(-75, 85, 10)
CMAP_GREY = cm.get_cmap("RdBu_r", len(BOUNDS_GREY) - 1)
NORM_GREY = colors.BoundaryNorm(BOUNDS_GREY, CMAP_GREY.N)
CMAP_GREY.set_bad("grey")
# %%
ut.check_make_dir(os.path.join(PATH_DATA, VERSION))
DF_CO_OCCURRENCES_FILENAME = os.path.join(PATH_DATA, VERSION, "df_co_occurrences_regions.csv")

df_clusters = pd.read_csv(
    os.path.join(PATH_CLUSTERS, VERSION, "df_clusters_full_ordered.csv"),
)
df_clusters["time"] = pd.to_datetime(df_clusters["time"])
df_clusters["time"] = df_clusters["time"].apply(
    lambda dt: dt.replace(hour=12, minute=0, second=0),
)  # set time to noon, to match df. Is daily average anyway

ed = pd.read_csv(
    os.path.join(
        "/usr/people/duinen/MSc-thesis/src/find_energydroughts/data/",
        "max_drought_regions_netto_demand_el7_winter_LENTIS_2023_PD_1600_events.csv",
    ),
).reset_index(drop=True)
ed["run"] = ed["runs"].str.extract(r"(\d+)").astype(int)
df_events = ed.drop(["Unnamed: 0", "runs"], axis=1)

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
    overlapping_events_list = []
    co_occur_prob_list = []

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

            co_occur_prob_list.append(
                {
                    "country_1": country_1,
                    "country_2": country_2,
                    "run": run,
                    "count": co_occur,
                },
            )

    overlapping_events_df = pd.DataFrame(overlapping_events_list)
    co_occur_prob_df = pd.DataFrame(co_occur_prob_list)

    co_occur_count_df = co_occur_prob_df.groupby(["country_1", "country_2"], as_index=False)[
        "count"
    ].sum()

    return overlapping_events_df, co_occur_prob_list, co_occur_count_df


def count_to_conditional(co_occur_count_df, country_order):
    pivot_table = (
        co_occur_count_df.pivot(index="country_1", columns="country_2", values="count")
        .reindex(country_order)
        .reindex(columns=country_order)
    )

    base_rates = np.diag(pivot_table).copy()

    conditional = pivot_table / base_rates[:, np.newaxis]
    return conditional, base_rates


def region_stats(probs_table, REGIONS):
    results = {}
    for region in REGIONS:
        region_countries = REGIONS[region]["countries"]
        region_combinations = list(product(region_countries, repeat=2))

        co_occurrences = np.array(
            [probs_table.loc[pair[0], pair[1]] for pair in region_combinations],
        )
        co_occurrences[co_occurrences == 1] = np.nan
        mean_co_occurrence = pd.Series(co_occurrences).mean()
        std_co_occurrence = pd.Series(co_occurrences).std()

        results[region] = {
            "mean_co_occurrence": mean_co_occurrence,
            "std_co_occurrence": std_co_occurrence,
        }

    return pd.DataFrame.from_dict(results, orient="index")


def monte_carlo(counts_per_run, n_sample, n_iter, vmax=None):
    count_per_run = pd.DataFrame(counts_per_run)
    conditional_probs = np.empty((n_iter, len(REGIONS), len(REGIONS)))

    for i in tqdm(range(n_iter)):
        random_runs = np.random.choice(np.arange(10, 170), size=n_sample, replace=False)
        selection_counts = count_per_run.query("run in @random_runs")
        co_occur_count_df = selection_counts.groupby(["country_1", "country_2"], as_index=False)[
            "count"
        ].sum()
        conditional, base_rates = count_to_conditional(
            co_occur_count_df,
            list(REGIONS.keys()),
        )
        conditional_probs[i] = conditional

    meanprob = np.mean(conditional_probs, axis=0)
    std = np.std(conditional_probs, axis=0)

    cg = sns.heatmap(
        (2 * std) / meanprob,
        cmap="Reds",
        xticklabels=list(REGIONS.keys()),
        yticklabels=list(REGIONS.keys()),
        vmax=vmax,
    )
    cbar = cg.collections[0].colorbar
    cbar.set_label("2*sigma/mean")
    return meanprob, std


def climatology_except_wr(co_occurrence_dict, i_sum):
    dataframes = []
    for i, wr in enumerate(wrs):
        if i == i_sum:
            continue
        df = pd.DataFrame(co_occurrence_dict[titles[i]]["counts_per_run"])
        df = df.groupby(df.columns.difference(["count"]).tolist()).sum().reset_index()
        dataframes.append(df)
    return dataframes[-1]


def p_value(meanprob, std, meanprob_all, std_all):
    SE = np.sqrt(std**2 + std_all**2)
    diff_means = meanprob - meanprob_all
    Z = diff_means / SE
    p_value = 2 * (1 - norm.cdf(abs(Z)))
    return p_value


# TODO: probably move these both to utils
def get_color(country):
    """Returns the color associated with a given country based on its regional classification dictionary."""
    for region in REGIONS.values():
        if country in region["countries"]:
            return region["color"]
    return None


def reorder(matrix):
    df_matrix = pd.DataFrame(matrix)

    matrix_reordered = df_matrix.reindex(ROW_ORDER, axis=0)
    matrix_reordered = matrix_reordered.reindex(ROW_ORDER, axis=1)
    return matrix_reordered.values


# %% CO-OCCURRENCE ANALYSIS
if not os.path.exists(DF_CO_OCCURRENCES_FILENAME):
    print("Computing co-occurrences...")
    events_df, counts_per_run, co_occur_counts_df = get_co_occurrence(df_full=df_events, wr="all")
    co_occur_counts_df.to_csv(DF_CO_OCCURRENCES_FILENAME, index=False)
    counts_per_run = pd.DataFrame(counts_per_run)
    counts_per_run.to_csv("counts_per_run_regions.csv")
else:
    print("Loading previously computed co-occurrences...")
    co_occur_counts_df = pd.read_csv(DF_CO_OCCURRENCES_FILENAME)
    counts_per_run = pd.read_csv("counts_per_run_regions.csv")

# %% MONTE CARLO SAMPLING OF CO-OCCURRENCE, FOR CONFIDENCE
meanprob, std = monte_carlo(counts_per_run, n_sample=100, n_iter=100)

# %%
# PLOT HEATMAP ALL WEATHER REGIMES, LINKAGE = DEFAULT AND COMPLETE
co_occur_counts_df["country_1_index"] = co_occur_counts_df["country_1"]
co_occur_counts_df["country_2_index"] = co_occur_counts_df["country_2"]
co_occur_counts_df["prob"] = co_occur_counts_df["count"] / 1600
co_occur_counts_df["prob"] = pd.to_numeric(co_occur_counts_df["prob"])

heatmap_data = co_occur_counts_df.pivot(
    index=["country_1_index"],
    columns=["country_2_index"],
    values="prob",
)

bounds = np.arange(0, 1.1, 0.1)

# Create a colormap with the specified boundaries
cmap = colors.ListedColormap(sns.color_palette("Reds", len(bounds) - 1))
norm_cmap = colors.BoundaryNorm(bounds, cmap.N)

# %%
plt.figure(figsize=(11, 8), dpi=300)
cg = sns.clustermap(
    heatmap_data.values,
    cmap=cmap,
    norm=norm_cmap,
    xticklabels=list(REGIONS.keys()),
    yticklabels=list(REGIONS.keys()),
)
cg.ax_heatmap.set_xticklabels(cg.ax_heatmap.get_xmajorticklabels(), fontsize=16)
cg.ax_heatmap.set_yticklabels(cg.ax_heatmap.get_ymajorticklabels(), fontsize=16)


cg.ax_col_dendrogram.set_visible(False)
x0, _y0, _w, _h = cg.cbar_pos
cg.ax_cbar.set_position([1.02, 0.06, 0.025, 0.74])
cg.cax.set_ylabel(r"P($E_{c,column}$|$E_{c,row}$)", size=18, rotation=90, labelpad=20)
cg.cax.yaxis.set_label_coords(x=3.5, y=0.5)
cg.cax.tick_params(labelsize=14)
cg.cax.yaxis.set_ticks(bounds)
row_order = cg.dendrogram_row.reordered_ind
region_order = np.array(list(REGIONS.keys()))[row_order]
# link = linkage(heatmap_data.values, method="complete")

# plt.figure(figsize=(11, 8))
# cg = sns.clustermap(
#     heatmap_data.values,
#     cmap="Reds",
#     xticklabels=COUNTRIES,
#     yticklabels=COUNTRIES,
#     row_linkage=link,
#     col_linkage=link,
# )
# cg.ax_heatmap.set_xticklabels(cg.ax_heatmap.get_xmajorticklabels(), fontsize=16)
# cg.ax_heatmap.set_yticklabels(cg.ax_heatmap.get_ymajorticklabels(), fontsize=16)


# cg.ax_col_dendrogram.set_visible(False)
# x0, _y0, _w, _h = cg.cbar_pos
# cg.ax_cbar.set_position([1.02, 0.06, 0.025, 0.74])
# cg.cax.set_ylabel(r"P($E_{c,column}$|$E_{c,row}$)", size=18, rotation=90, labelpad=20)
# cg.cax.yaxis.set_label_coords(x=3.5, y=0.5)
# cg.cax.tick_params(labelsize=14)
# row_order = cg.dendrogram_row.reordered_ind
# %%
co_occurrence_all = reorder(heatmap_data.values)

# %%
plt.figure(figsize=(11, 8), dpi=300)
cg = sns.heatmap(
    meanprob,
    cmap=cmap,
    norm=norm_cmap,
    xticklabels=list(REGIONS.keys()),
    yticklabels=list(REGIONS.keys()),
    cbar=False,
)
cg.set_xticklabels(cg.get_xmajorticklabels(), fontsize=16, rotation=90)
cg.set_yticklabels(cg.get_ymajorticklabels(), fontsize=16)
cbar = plt.colorbar(cg.collections[0])
cbar.set_label(r"P($E_{c,column}$|$E_{c,row}$)", size=18)
# %%
# REPEAT CALCULATION FOR EACH DISTINCT WEATHER REGIME
wrs = np.arange(6)
titles = CLUSTER_NAMES + ["No regime", "No persisting regime"]

if not os.path.exists(os.path.join(PATH_DATA, VERSION, "co_occurrence_dict_regions.pickle")):
    print("Computing co-occurrences per weather regime...")

    co_occurrence_dict = {}

    for i, wr in enumerate(wrs):
        print(f"Computing co-occurrences for {titles[i]}...")
        events_df_wr, counts_per_run, counts_df_wr = get_co_occurrence(df_full=df_events, wr=wr)

        probs_table, base_rates = count_to_conditional(counts_df_wr, list(REGIONS.keys()))

        co_occurrence_dict[titles[i]] = {
            "events_df": events_df_wr,
            "counts_per_run": counts_per_run,
            "counts_df": counts_df_wr,
            "probs_table": probs_table,
            "base_rates": base_rates,
        }

        # quick visualization
        plt.figure()
        sns.heatmap(probs_table, cmap="Reds")
        plt.title(titles[i])
        plt.show()

        monte_carlo(counts_per_run, n_sample=100, n_iter=100)

    # write
    with open(os.path.join(PATH_DATA, VERSION, "co_occurrence_dict_regions.pickle"), "wb") as f:
        pickle.dump(co_occurrence_dict, f)
else:
    print("Loading previously computed co-occurrences...")
    co_occurrence_dict = pickle.load(
        open(os.path.join(PATH_DATA, VERSION, "co_occurrence_dict_regions.pickle"), "rb"),
    )

# %% MONTE CARLO PER WR WITH P-TEST
for i, wr in enumerate(wrs):
    plt.figure()
    meanprob, std = monte_carlo(
        co_occurrence_dict[titles[i]]["counts_per_run"],
        n_sample=100,
        n_iter=100,
        vmax=0.25,
    )
    plt.title(titles[i])
    plt.show()

    plt.figure()
    meanprob_all, std_all = monte_carlo(
        climatology_except_wr(co_occurrence_dict, i),
        n_sample=100,
        n_iter=100,
        vmax=0.25,
    )
    plt.title(titles[i])
    plt.show()

    diff = (meanprob - meanprob_all) / meanprob_all
    p_val = p_value(meanprob, std, meanprob_all, std_all)

    masked_diff_significant = np.where(p_val > 0.05, np.nan, diff)
    signal_to_noise = meanprob / (2 * std)
    masked_diff = np.where(signal_to_noise < 5, np.nan, masked_diff_significant)

    if np.logical_and(
        (100 * np.nanmax(masked_diff) > np.nanmax(BOUNDS_GREY)),
        (100 * np.nanmin(masked_diff) < np.nanmin(BOUNDS_GREY)),
    ):
        extend = "both"
    elif 100 * np.nanmax(masked_diff) > np.nanmax(BOUNDS_GREY):
        extend = "max"
    elif 100 * np.nanmin(masked_diff) < np.nanmin(BOUNDS_GREY):
        extend = "min"
    else:
        extend = "neither"

    plt.figure(dpi=300)
    cg = sns.heatmap(
        100 * masked_diff,
        vmin=-75,
        vmax=75,
        cmap=CMAP_GREY,
        norm=NORM_GREY,
        cbar=False,
        xticklabels=list(REGIONS.keys()),
        yticklabels=list(REGIONS.keys()),
    )
    cbar = plt.colorbar(cg.collections[0], extend=extend)
    cbar.set_label("Probability difference [%]")
    tick_vals = np.arange(-75, 76, 10)
    cbar.set_ticks(tick_vals)
    cbar.set_ticklabels(tick_vals)
    plt.title(titles[i])
    plt.show()

# %%
# plot co-occurrence probabilities per WR, and calc stats per region

stats_dict = {}

for i, wr in enumerate(wrs):
    probs_table = co_occurrence_dict[titles[i]]["probs_table"]

    risk_ratio = (co_occurrence_dict[titles[i]]["base_rates"] / 1600) / cluster_occurrences[i]

    plt.figure(figsize=(12, 8))
    gs = gridspec.GridSpec(1, 2, width_ratios=[1, 6])
    ax_bar = plt.subplot(gs[0])
    ax_heatmap = plt.subplot(gs[1])
    num_heatmap_rows = probs_table.values.shape[0]

    bar_widths = np.abs(risk_ratio - 1)
    bar_starts = np.where(risk_ratio < 1, risk_ratio, 1)

    for j, country in enumerate(np.array(COUNTRIES)[ROW_ORDER]):
        left = bar_starts[j]
        width = bar_widths[j]
        ax_bar.barh(
            y=country,
            width=width,
            height=0.6,
            left=left,
            color=get_color(country),
            edgecolor="black",
        )

    ax_bar.set_ylim(-0.5, num_heatmap_rows - 0.5)
    ax_bar.invert_yaxis()
    ax_bar.set(yticklabels=[], yticks=[])
    ax_bar.set_xscale("log")
    ax_bar.set_xticks([0.2, 0.5, 1, 2, 5], [0.2, 0.5, 1, 2, 5])
    if i == 0:
        ax_bar.set_xlim(0.01, 5)
    else:
        ax_bar.set_xlim(0.1, 5)
    ax_bar.set_xlabel("Risk ratio")
    xaxis_line = ax_bar.get_xaxis().get_gridlines()[0]
    line_width = xaxis_line.get_linewidth()
    ax_bar.axvline(x=1, color="k", linestyle="-", linewidth=line_width)

    sns.heatmap(probs_table, cmap="Reds", vmin=0, vmax=1, cbar=True)
    ax_heatmap.set(xlabel="", ylabel="")
    cbar = ax_heatmap.collections[0].colorbar
    plt.title(titles[i])
    plt.show()

    stats_dict[titles[i]] = region_stats(probs_table, REGIONS)
# %%
# calculate and plot differences in co-occurrence probabilities

masked_co_occurrence_all = np.where(co_occurrence_all < 0.3, np.nan, co_occurrence_all)
masked_co_occurrence_all = np.where(co_occurrence_all == 1, np.nan, masked_co_occurrence_all)
for i, wr in enumerate(wrs):
    probs_table = co_occurrence_dict[titles[i]]["probs_table"]
    diff = 100 * (probs_table - masked_co_occurrence_all) / co_occurrence_all

    risk_ratio = (co_occurrence_dict[titles[i]]["base_rates"] / 1600) / cluster_occurrences[i]

    plt.figure(figsize=(12, 8))
    gs = gridspec.GridSpec(1, 2, width_ratios=[1, 6])
    ax_bar = plt.subplot(gs[0])
    ax_heatmap = plt.subplot(gs[1])
    num_heatmap_rows = diff.shape[0]

    bar_widths = np.abs(risk_ratio - 1)
    bar_starts = np.where(risk_ratio < 1, risk_ratio, 1)

    for j, country in enumerate(np.array(COUNTRIES)[ROW_ORDER]):
        left = bar_starts[j]
        width = bar_widths[j]
        ax_bar.barh(
            y=country,
            width=width,
            height=0.6,
            left=left,
            color=get_color(country),
            edgecolor="black",
        )

    ax_bar.set_ylim(-0.5, num_heatmap_rows - 0.5)
    ax_bar.invert_yaxis()
    ax_bar.set(yticklabels=[], yticks=[])
    ax_bar.set_xscale("log")
    ax_bar.set_xticks([0.2, 0.5, 1, 2, 5], [0.2, 0.5, 1, 2, 5])
    if i == 0:
        ax_bar.set_xlim(0.01, 5)
    else:
        ax_bar.set_xlim(0.1, 5)
    ax_bar.set_xlabel("Risk ratio")
    xaxis_line = ax_bar.get_xaxis().get_gridlines()[0]
    line_width = xaxis_line.get_linewidth()
    ax_bar.axvline(x=1, color="k", linestyle="-", linewidth=line_width)

    sns.heatmap(diff, cmap=CMAP_GREY, vmin=-50, vmax=50, cbar=True)
    ax_heatmap.set(xlabel="", ylabel="")
    cbar = ax_heatmap.collections[0].colorbar
    cbar.set_label("[%]")
    plt.title(f"Co-occurrence anomalies for {titles[i]} vs. climatology")
    plt.show()

# %%
# RECLUSTERING COMPLETE FOR EVERY DISTINCT WEATHER REGIME
for i, wr in enumerate(wrs):
    probs_table = co_occurrence_dict[titles[i]]["probs_table"]

    link = linkage(probs_table.values, method="complete")
    plt.figure(figsize=(12, 8))

    cg = sns.clustermap(
        probs_table.values,
        cmap="Reds",
        xticklabels=np.asarray(COUNTRIES)[ROW_ORDER],
        yticklabels=np.asarray(COUNTRIES)[ROW_ORDER],
        row_linkage=link,
        col_linkage=link,
    )
    cg.ax_heatmap.set_xticklabels(cg.ax_heatmap.get_xmajorticklabels(), fontsize=16)
    cg.ax_heatmap.set_yticklabels(cg.ax_heatmap.get_ymajorticklabels(), fontsize=16)

    cg.ax_heatmap.set_title(titles[i], fontsize=20)

    cg.ax_col_dendrogram.set_visible(False)
    x0, _y0, _w, _h = cg.cbar_pos
    cg.ax_cbar.set_position([1.02, 0.06, 0.025, 0.74])
    cg.cax.set_ylabel(r"P($E_{c,column}$|$E_{c,row}$)", size=18, rotation=90, labelpad=20)
    cg.cax.yaxis.set_label_coords(x=3.5, y=0.5)
    cg.cax.tick_params(labelsize=14)


# %% RECLUSTERING INTO PRECISELY 6 CLUSTERS AND CLUSTER_STATS
# EXPERIMENTAL, NOT SURE IF IT IS A GOOD IDEA


def calc_cluster_stats(clusters):
    cluster_stats = []

    for cluster_id in range(1, 7):  # Assuming cluster IDs are 1 through 6
        # Find indices of countries in the current cluster
        indices = np.where(clusters == cluster_id)[0]

        # Extract the sub-matrix for the current cluster
        cluster_data = probs_table.values[np.ix_(indices, indices)]

        # Exclude the diagonal elements
        non_diagonal_values = cluster_data[~np.eye(cluster_data.shape[0], dtype=bool)]

        # Calculate mean and standard deviation
        mean_value = np.mean(non_diagonal_values)
        std_deviation = np.std(non_diagonal_values)

        # Store results for the cluster
        cluster_stats.append((mean_value, std_deviation))

    # Display results
    for idx, (mean, std) in enumerate(cluster_stats):
        print(f"Cluster {idx + 1}: Mean = {mean:.4f}, Std Dev = {std:.4f}")

    for i, country in enumerate(reordered_countries):
        print(country, clusters[order][i])


# Define cluster colors
colors_clusters = ["C0", "C1", "C2", "C3", "C4", "cyan"]

for i, wr in enumerate(wrs):
    probs_table = co_occurrence_dict[titles[i]]["probs_table"]

    # Create linkage matrix
    link = linkage(probs_table.values, method="complete")

    # Determine the clusters, cutting the dendrogram at 6 clusters
    clusters = fcluster(link, t=6, criterion="maxclust")

    # Order the rows and columns according to the clusters
    order = np.argsort(clusters)
    reordered_data = probs_table.values[order][:, order]
    reordered_countries = np.asarray(COUNTRIES)[ROW_ORDER][order]

    # Create new linkage matrix for the reordered data
    reordered_link = linkage(reordered_data, method="complete")

    plt.figure(figsize=(12, 8))

    cg = sns.clustermap(
        reordered_data,
        cmap="Reds",
        xticklabels=reordered_countries,
        yticklabels=reordered_countries,
        row_linkage=reordered_link,
        col_linkage=reordered_link,
    )

    # Apply color to each tick label according to its cluster
    for ax in [cg.ax_heatmap.get_xaxis(), cg.ax_heatmap.get_yaxis()]:
        labels = ax.get_majorticklabels()
        for label in labels:
            idx = np.where(reordered_countries == label.get_text())[0][0]
            label.set_color(colors_clusters[clusters[order][idx] - 1])

    cg.ax_heatmap.set_xticklabels(cg.ax_heatmap.get_xmajorticklabels(), fontsize=16)
    cg.ax_heatmap.set_yticklabels(cg.ax_heatmap.get_ymajorticklabels(), fontsize=16)

    cg.ax_heatmap.set_title(titles[i], fontsize=20)

    cg.ax_col_dendrogram.set_visible(False)
    x0, _y0, _w, _h = cg.cbar_pos
    cg.ax_cbar.set_position([1.02, 0.06, 0.025, 0.74])
    cg.cax.set_ylabel(r"P($E_{c,column}$|$E_{c,row}$)", size=18, rotation=90, labelpad=20)
    cg.cax.yaxis.set_label_coords(x=3.5, y=0.5)
    cg.cax.tick_params(labelsize=14)

    plt.show()

    calc_cluster_stats(clusters)

# %%
