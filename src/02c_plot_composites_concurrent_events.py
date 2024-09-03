"""Script to plot composite meteorology for concurrent energy drought events in two countries

Author: Bram van Duinen (bramvduinen@gmail.com)

"""

# %%
import os

import cartopy
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import utils as ut
import xarray as xr
from config import ED_FILENAME_COUNTRIES, PATH_ED
from matplotlib import colors
from tqdm.notebook import tqdm

# %%
ed = pd.read_csv(
    os.path.join(PATH_ED, ED_FILENAME_COUNTRIES),
    #    os.path.join(PATH_ED_REGIONS, ED_FILENAME_REGIONS), # for regions instead of countries
).reset_index(drop=True)

ed["run"] = ed["runs"].str.extract(r"(\d+)").astype(int)
df_events = ed.drop(["Unnamed: 0", "runs"], axis=1)

ANOM_PATH = "/net/pc230050/nobackup/users/duinen/LENTIS/present"
PSL_PATH = "/net/pc200256/nobackup/users/most/LENTIS/present/day/psl_d"
dir_Figures = (
    "/usr/people/duinen/MSc-thesis/Results/Figures/composites_events/PD_2023_1600_multiplecountries"
)
dir_Data = "/net/pc230050/nobackup/users/duinen/results/event_anomalies_per_country/PD_2023_1600_multiplecountries"

ut.check_make_dir(dir_Figures)
ut.check_make_dir(dir_Data)

dataproj = ccrs.EuroPP()
cmap = plt.cm.RdBu_r

GyYl = colors.LinearSegmentedColormap.from_list(
    "my_colormap",
    [(0, "grey"), (0.5, "white"), (1, "orange")],  # (position, color)
)


# %%
def create_composite_dataset(df_events):
    """Create composite dataset with meteorological variables for each country during its energy drought events"""
    composite_datasets_by_country = {}
    countries = df_events.country.unique()
    for run in tqdm(np.arange(10, 170)):
        df_run = df_events.query("run == @run").sort_values(["country", "start_time"])

        rsds_run = xr.open_dataset(
            f"{ANOM_PATH}/rsds_d_anomaly/anom_rsds_d_ECEarth3_h{run:03d}.nc",
        )["rsds"]
        sfcWind_run = xr.open_dataset(
            f"{ANOM_PATH}/sfcWind_d_anomaly/anom_sfcWind_d_ECEarth3_h{run:03d}.nc",
        )["sfcWind"].drop_vars("height")
        t2m_run = xr.open_dataset(
            f"{ANOM_PATH}/tas_d_anomaly/anom_tas_d_ECEarth3_h{run:03d}.nc",
        )["tas"].drop_vars("height")
        psl_run = xr.open_dataset(f"{PSL_PATH}/psl_d_ECEarth3_h{run:03d}.nc")["psl"]

        for country in countries:
            times_run = []
            for event, row in df_run.query("country == @country").iterrows():
                timeseries_event = pd.date_range(
                    start=row["start_time"],
                    end=row["end_time"],
                    freq="D",
                )
                times_run.extend(timeseries_event)

            rsds_subset = rsds_run.sel(time=times_run)
            sfcWind_subset = sfcWind_run.sel(time=times_run)
            t2m_subset = t2m_run.sel(time=times_run)
            psl_subset = psl_run.sel(time=times_run)

            temp_dataset = xr.Dataset(
                {
                    "rsds": rsds_subset,
                    "sfcWind": sfcWind_subset,
                    "tas": t2m_subset,
                    "psl": psl_subset,
                },
                coords={
                    "time": times_run,
                    "lon": rsds_subset.lon,
                    "lat": rsds_subset.lat,
                    "country": country,
                    "run": run,
                },
            )

            if country not in composite_datasets_by_country:
                composite_datasets_by_country[country] = []
            composite_datasets_by_country[country].append(temp_dataset)

    return composite_datasets_by_country


def find_streak(days):
    streaks = []
    current_streak = []

    for day in sorted(days):
        if not current_streak or day == current_streak[-1] + pd.Timedelta(days=1):
            current_streak.append(day)
        else:
            # If the streak is at least 7 days, append it
            if len(current_streak) >= 7:
                streaks.extend(current_streak)
            current_streak = [day]

    # Check the last streak
    if len(current_streak) >= 7:
        streaks.extend(current_streak)

    return streaks


def find_overlapping_times(composite_datasets_by_country, country_combination):
    """Find overlapping and disjoint times for energy drought events in two countries"""
    overlapping_times_per_run = {}
    unique_c1_per_run = {}
    unique_c2_per_run = {}

    for run in np.arange(10, 170):
        times_country_1 = composite_datasets_by_country[country_combination[0]][
            run - 10
        ].time.values
        times_country_2 = composite_datasets_by_country[country_combination[1]][
            run - 10
        ].time.values

        common_times = set(times_country_1).intersection(set(times_country_2))
        unique_c1 = set(times_country_1).difference(set(times_country_2))
        unique_c2 = set(times_country_2).difference(set(times_country_1))

        overlapping_times_per_run[run] = sorted(list(common_times))
        unique_c1_per_run[run] = find_streak(unique_c1)
        unique_c2_per_run[run] = find_streak(unique_c2)

    return overlapping_times_per_run, unique_c1_per_run, unique_c2_per_run


def meteo_overlapping_countries(country_combination):
    """Create dataset with meteorological variables during overlapping and disjoint energy drought events"""
    overlapping_datasets = []
    c1_unique_datasets = []
    c2_unique_datasets = []
    times_overlap, times_c1, times_c2 = find_overlapping_times(
        composite_datasets_by_country,
        country_combination,
    )

    for ds in composite_datasets_by_country[country_combination[0]]:
        run_nr = int(ds.run.values)
        if times_overlap[run_nr]:
            overlapping_dataset_temp = xr.Dataset(
                {
                    "rsds": ds.rsds.sel(time=times_overlap[run_nr]),
                    "sfcWind": ds.sfcWind.sel(time=times_overlap[run_nr]),
                    "tas": ds.tas.sel(time=times_overlap[run_nr]),
                    "psl": ds.psl.sel(time=times_overlap[run_nr]),
                },
            )
            overlapping_datasets.append(overlapping_dataset_temp)

        if times_c1[run_nr]:
            c1_unique_dataset_temp = xr.Dataset(
                {
                    "rsds": ds.rsds.sel(time=times_c1[run_nr]),
                    "sfcWind": ds.sfcWind.sel(time=times_c1[run_nr]),
                    "tas": ds.tas.sel(time=times_c1[run_nr]),
                    "psl": ds.psl.sel(time=times_c1[run_nr]),
                },
            )

            c1_unique_datasets.append(c1_unique_dataset_temp)

    for ds in composite_datasets_by_country[country_combination[1]]:
        run_nr = int(ds.run.values)
        if times_c2[run_nr]:
            c2_unique_dataset_temp = xr.Dataset(
                {
                    "rsds": ds.rsds.sel(time=times_c2[run_nr]),
                    "sfcWind": ds.sfcWind.sel(time=times_c2[run_nr]),
                    "tas": ds.tas.sel(time=times_c2[run_nr]),
                    "psl": ds.psl.sel(time=times_c2[run_nr]),
                },
            )
            c2_unique_datasets.append(c2_unique_dataset_temp)
    return overlapping_datasets, c1_unique_datasets, c2_unique_datasets


def plot_composite_meteorology_countrycomb(
    country_combination,
    overlapping_datasets,
    c1_unique_dataset,
    c2_unique_dataset,
):
    """Plot composite meteorology for overlapping, country 1 and country 2 disjoint datasets"""
    file_path_overlapping = f"{dir_Data}/{country_combination}_concurrent_ds.nc"
    if os.path.exists(file_path_overlapping):
        print("Overlapping dataset already exists")
        ds_overlapping = xr.open_dataset(file_path_overlapping)
    else:
        ds_overlapping = xr.concat(overlapping_datasets, dim="time")
        ds_overlapping.to_netcdf(file_path_overlapping)

    file_path_c1 = f"{dir_Data}/{country_combination}_c1_ds.nc"
    if os.path.exists(file_path_c1):
        print("Country 1 dataset already exists")
        ds_c1 = xr.open_dataset(file_path_c1)
    else:
        ds_c1 = xr.concat(c1_unique_dataset, dim="time")
        ds_c1.to_netcdf(file_path_c1)

    file_path_c2 = f"{dir_Data}/{country_combination}_c2_ds.nc"
    if os.path.exists(file_path_c2):
        print("Country 2 dataset already exists")
        ds_c2 = xr.open_dataset(file_path_c2)
    else:
        ds_c2 = xr.concat(c2_unique_dataset, dim="time")
        ds_c2.to_netcdf(file_path_c2)

    lons = ds_overlapping.lon.values
    lats = ds_overlapping.lat.values

    fig, axs = plt.subplots(
        3,
        3,
        figsize=(12, 17),
        subplot_kw={"projection": ccrs.EuroPP()},
        dpi=300,
    )
    fontsize = 15
    fig.subplots_adjust(hspace=0.15, wspace=0.1, top=0.95, bottom=0.1, left=0.05, right=0.95)

    vmin_list = [-7, -5, -30]
    vmax_list = [7, 5, 30]
    alphabet = ["a", "b", "c", "d", "e", "f", "g", "h", "i"]
    titles = ["2m temperature", "10 m wind speed", "Solar radiation"]
    cmaps = [plt.cm.RdBu_r, plt.cm.PiYG, GyYl]
    cbar_units = ["[K]", "[m/s]", "[W/m²]"]
    data_column1 = [
        (
            ds_c2.psl.mean(dim="time") / 100,
            ds_c2.tas.mean(dim="time"),
            ds_c2.sfcWind.mean(dim="time"),
            ds_c2.rsds.mean(dim="time"),
        ),
        (
            ds_c1.psl.mean(dim="time") / 100,
            ds_c1.tas.mean(dim="time"),
            ds_c1.sfcWind.mean(dim="time"),
            ds_c1.rsds.mean(dim="time"),
        ),
        (
            ds_overlapping.psl.mean(dim="time") / 100,
            ds_overlapping.tas.mean(dim="time"),
            ds_overlapping.sfcWind.mean(dim="time"),
            ds_overlapping.rsds.mean(dim="time"),
        ),
    ]

    for i in range(3):
        for j in range(3):
            ax = axs[i, j]
            data = data_column1[j][i + 1]
            psl = data_column1[j][0]

            levels = np.linspace(vmin_list[i], vmax_list[i], 11)
            norm = colors.BoundaryNorm(levels, ncolors=cmaps[i].N, extend="both")
            im = ax.contourf(
                lons,
                lats,
                data,
                transform=ccrs.PlateCarree(),
                levels=levels,
                cmap=cmaps[i],
                norm=norm,
                extend="both",
            )
            CS = ax.contour(
                lons,
                lats,
                psl,
                transform=ccrs.PlateCarree(),
                colors="k",
            )

            ax.clabel(CS, inline=True, fontsize=10)

            ax.set_extent([-30, 33, 35, 70], crs=ccrs.PlateCarree())
            ax.coastlines()
            ax.add_feature(cartopy.feature.BORDERS, linestyle=":", alpha=1)

            if i == 0:
                if j == 0:
                    ax.set_title(f"{country_combination[1]} disjoint", fontsize=fontsize)
                elif j == 1:
                    ax.set_title(f"{country_combination[0]} disjoint", fontsize=fontsize)
                else:
                    ax.set_title("Co-occurrence", fontsize=fontsize)

            ax.set_title(f"({alphabet[3*i+j]})", loc="left", fontsize=fontsize)

            if j == 0:
                ax.text(
                    -0.1,
                    0.55,
                    titles[i],
                    va="bottom",
                    ha="center",
                    rotation="vertical",
                    rotation_mode="anchor",
                    transform=ax.transAxes,
                    fontsize=fontsize,
                )

        # Adding colorbars
        if j == 2:
            cbar_ax = fig.add_axes([0.2, 0.08 + (0.3 * (2 - i)), 0.6, 0.02])
            cb = fig.colorbar(im, cax=cbar_ax, orientation="horizontal")
            cb.set_label(cbar_units[i], fontsize=fontsize)
            cb.ax.tick_params(labelsize=fontsize)

    plt.show()


# %%
composite_datasets_by_country = create_composite_dataset(df_events)

# %%
europe_neighbors = {
    "AUT": ["DEU", "CZE", "SVK", "HUN", "SVN", "ITA", "CHE"],
    "BEL": ["NLD", "DEU", "FRA", "GBR"],
    "CHE": ["DEU", "AUT", "FRA", "ITA"],
    "CZE": ["DEU", "AUT", "SVK", "POL"],
    "DEU": [
        "DNK",
        "NLD",
        "BEL",
        "FRA",
        "CHE",
        "AUT",
        "CZE",
        "POL",
        "GBR",
        "CHE",
        "NOR",
        "SWE",
    ],
    "DNK": ["DEU", "SWE", "NOR", "GBR"],
    "ESP": ["FRA", "PRT"],
    "EST": ["FIN", "LVA"],
    "FIN": ["SWE", "EST", "NOR"],
    "FRA": ["BEL", "DEU", "CHE", "ITA", "ESP", "GBR", "IRL"],
    "GBR": ["IRL", "FRA", "BEL", "NLD", "DEU", "DNK", "NOR"],
    "HRV": ["SVN", "HUN", "ITA"],
    "HUN": ["AUT", "SVK", "HRV", "SVN"],
    "IRL": ["GBR", "FRA"],
    "ITA": ["FRA", "CHE", "AUT", "SVN", "HRV"],
    "LTU": ["LVA", "POL", "SWE"],
    "LVA": ["EST", "LTU", "SWE"],
    "NLD": ["BEL", "DEU", "GBR", "DNK", "NOR"],
    "NOR": ["SWE", "DNK", "GBR", "NLD", "DEU", "FIN"],
    "POL": ["DEU", "CZE", "SVK", "LTU", "SWE"],
    "PRT": ["ESP"],
    "SVK": ["POL", "CZE", "AUT", "HUN"],
    "SVN": ["AUT", "ITA", "HUN", "HRV"],
    "SWE": ["NOR", "FIN", "DNK", "DEU", "POL", "LTU", "LVA"],
}

country_combinations = [["DEU", "POL"]]  # only one of interest now, to plot for paper

# uncomment lines below to plot all possible country combinations
# country_combinations = []
# for key, values in europe_neighbors.items():
#     for value in values:
#         pair = [key, value]
#         reverse_pair = [value, key]
#         if pair not in country_combinations and reverse_pair not in country_combinations:
#             country_combinations.append(pair)

for country_combination in tqdm(country_combinations):
    print(f"Starting {country_combination}...")
    overlapping_datasets, c1_unique_datasets, c2_unique_datasets = meteo_overlapping_countries(
        country_combination,
    )
    plot_composite_meteorology_countrycomb(
        country_combination,
        overlapping_datasets,
        c1_unique_datasets,
        c2_unique_datasets,
    )

# %%
