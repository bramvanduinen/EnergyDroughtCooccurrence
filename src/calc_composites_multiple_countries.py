# %%
import utils as ut
import plot_utils as put

import os
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.colors import LinearSegmentedColormap
import cartopy.crs as ccrs
import cartopy
from tqdm.notebook import tqdm
from matplotlib import ticker
from matplotlib.cm import RdBu_r

# %%
# PATH_ED = '../../energydroughts-Europe/data/'
PATH_ED = "/usr/people/duinen/energydroughts-Europe/data/"
ed = pd.read_csv(
    os.path.join(PATH_ED, "netto_demand_el7_winter_LENTIS_2023_PD_1600_events.csv")
).reset_index(drop=True)
ed["run"] = ed["runs"].str.extract("(\d+)").astype(int)
df_events = ed.drop(["Unnamed: 0", "runs"], axis=1)

# %%
ANOM_PATH = "/net/pc230050/nobackup/users/duinen/LENTIS/present"
PSL_PATH = "/net/pc200256/nobackup/users/most/LENTIS/present/day/psl_d"
dir_Figures = "/usr/people/duinen/MSc-thesis/Results/Figures/composites_events/PD_2023_1600_multiplecountries"
dir_Data = "/net/pc230050/nobackup/users/duinen/results/event_anomalies_per_country/PD_2023_1600_multiplecountries"
NUM_EVENTS = 1600
# %%
ut.check_make_dir(dir_Figures)
ut.check_make_dir(dir_Data)


# %%
def create_composite_dataset(df_events):
    composite_datasets_by_country = {}
    countries = df_events.country.unique()
    for run in tqdm(np.arange(10, 170)):
        df_run = df_events.query("run == @run").sort_values(["country", "start_time"])

        rsds_run = xr.open_dataset(
            f"{ANOM_PATH}/rsds_d_anomaly/anom_rsds_d_ECEarth3_h{run:03d}.nc"
        )["rsds"]
        sfcWind_run = xr.open_dataset(
            f"{ANOM_PATH}/sfcWind_d_anomaly/anom_sfcWind_d_ECEarth3_h{run:03d}.nc"
        )["sfcWind"].drop_vars("height")
        t2m_run = xr.open_dataset(
            f"{ANOM_PATH}/tas_d_anomaly/anom_tas_d_ECEarth3_h{run:03d}.nc"
        )["tas"].drop_vars("height")
        psl_run = xr.open_dataset(f"{PSL_PATH}/psl_d_ECEarth3_h{run:03d}.nc")["psl"]

        for country in countries:  # df_run.country.unique():
            times_run = []
            for event, row in df_run.query("country == @country").iterrows():
                timeseries_event = pd.date_range(
                    start=row["start_time"], end=row["end_time"], freq="D"
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


def find_overlapping_times(composite_datasets_by_country, country_combination):
    overlapping_times_per_run = {}
    unique_c1_per_run = {}
    unique_c2_per_run = {}

    # count_c1 = 0  # TEMPORARY
    # count_c2 = 0  # TEMPORARY
    for run in np.arange(10, 170):

        common_times = None
        times_country_1 = composite_datasets_by_country[country_combination[0]][
            run - 10
        ].time.values
        times_country_2 = composite_datasets_by_country[country_combination[1]][
            run - 10
        ].time.values

        common_times = set(times_country_1).intersection(set(times_country_2))
        unique_c1 = set(times_country_1).difference(set(times_country_2))
        unique_c2 = set(times_country_2).difference(set(times_country_1))

        overlapping_times_per_run[run] = {}
        unique_c1_per_run[run] = {}
        unique_c2_per_run[run] = {}

        if common_times:
            overlapping_times_per_run[run] = sorted(list(common_times))
        if unique_c1:
            unique_c1_per_run[run] = sorted(list(unique_c1))
            # count_c1 += len(unique_c1)
        if unique_c2:
            unique_c2_per_run[run] = sorted(list(unique_c2))
            # count_c2 += len(unique_c2)

    # print(f"Unique c1: {count_c1}, unique c2: {count_c2}")

    return overlapping_times_per_run, unique_c1_per_run, unique_c2_per_run


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


def find_overlapping_times_v2(composite_datasets_by_country, country_combination):
    overlapping_times_per_run = {}
    unique_c1_per_run = {}
    unique_c2_per_run = {}

    # count_c1 = 0
    # count_c2 = 0
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
        # count_c1 += len(find_streak(unique_c1))
        unique_c2_per_run[run] = find_streak(unique_c2)
        # count_c2 += len(find_streak(unique_c2))

    # print(f"Unique c1: {count_c1}, unique c2: {count_c2}")

    return overlapping_times_per_run, unique_c1_per_run, unique_c2_per_run


def meteo_overlapping_countries(country_combination):
    overlapping_datasets = []
    c1_unique_datasets = []
    c2_unique_datasets = []
    times_overlap, times_c1, times_c2 = find_overlapping_times_v2(
        composite_datasets_by_country, country_combination
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


# %%
# dataproj = ccrs.PlateCarree()
dataproj = ccrs.LambertConformal(central_longitude=0, central_latitude=45)
cmap = plt.cm.RdBu_r


def plot_composite_meteorology(country, composite_datasets_by_country):
    file_path = f"{dir_Data}/{country}_ds_events.nc"
    if os.path.exists(file_path):
        print("Already exists")
        ds_country = xr.open_dataset(file_path)
    else:
        ds_country = xr.concat(composite_datasets_by_country[country], dim="time")
        ds_country.to_netcdf(file_path)

    lons = ds_country.lon.values
    lats = ds_country.lat.values
    lons_plot = lons[(lons < 33) & (lons > -30)]
    lats_plot = lats[(lats < 70) & (lats > 35)]

    psl_country = ds_country.psl.mean(dim="time")
    tas_country = ds_country.tas.mean(dim="time")
    sfcWind_country = ds_country.sfcWind.mean(dim="time")
    rsds_country = ds_country.rsds.mean(dim="time")

    fig, axs = plt.subplots(3, 1, figsize=(6, 10), subplot_kw={"projection": dataproj})
    fig.subplots_adjust(right=0.8)

    vmin_list = [-5, -5, -30]
    vmax_list = [5, 5, 30]

    data = [tas_country, sfcWind_country, rsds_country]
    titles = ["Temperature", "Wind speed", "Radiation"]
    cbar_units = ["[K]", "[m/s]", "[W/m²]"]
    lon_lim, lat_lim = 33, 70

    for i in range(3):
        anom_plot = data[i]
        levels = np.linspace(vmin_list[i], vmax_list[i], 11)
        ax = axs.flat[i]
        psl_plot = psl_country / 100  # Pa to hPa
        psl_plothl = psl_plot.where(
            (psl_plot.lon > -30)
            & (psl_plot.lon < 33)
            & (psl_plot.lat > 35)
            & (psl_plot.lat < 70),
            drop=True,
        )
        ax.set_extent([-30, 33, 35, 70], crs=ccrs.PlateCarree())

        norm = colors.BoundaryNorm(levels, ncolors=cmap.N, extend="both")
        im = ax.contourf(
            lons,
            lats,
            anom_plot,
            transform=ccrs.PlateCarree(),
            levels=levels,
            cmap=cmap,
            norm=norm,
            extend="both",
        )
        fig.colorbar(im, ax=ax, label=cbar_units[i])
        CS = ax.contour(lons, lats, psl_plot, transform=ccrs.PlateCarree(), colors="k")
        ax.clabel(CS, inline=True, fontsize=10)
        put.plot_maxmin_points(
            ax,
            lons_plot,
            lats_plot,
            lon_lim,
            lat_lim,
            psl_plothl,
            "max",
            50,
            "H",
            color="k",
            plotValue=False,
            transform=ccrs.PlateCarree(),
        )
        put.plot_maxmin_points(
            ax,
            lons_plot,
            lats_plot,
            lon_lim,
            lat_lim,
            psl_plothl,
            "min",
            50,
            "L",
            color="k",
            plotValue=False,
            transform=ccrs.PlateCarree(),
        )
        ax.coastlines()
        ax.add_feature(cartopy.feature.BORDERS, linestyle=":", alpha=1)
        gl = ax.gridlines(
            crs=ccrs.PlateCarree(),
            draw_labels=True,
            x_inline=False,
            rotate_labels=False,
        )
        gl.top_labels = False
        gl.right_labels = False
        gl.xlocator = ticker.FixedLocator([-20, -10, 0, 10, 20, 30])
        ax.set_title(titles[i])
    fig.suptitle(f"Composite anomalies during {country} events")
    plt.tight_layout()
    plt.savefig(f"{dir_Figures}/{country}_composite_meteorology.png", dpi=300)
    ds_country.close()


GyYl = colors.LinearSegmentedColormap.from_list(
    "my_colormap",
    [(0, "grey"), (0.5, "white"), (1, "yellow")],  # (position, color)
)


# %%
def plot_composite_meteorology_countrycomb(
    country_combination, overlapping_datasets, c1_unique_dataset, c2_unique_dataset
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

    lons_plot = lons[(lons < 33) & (lons > -30)]
    lats_plot = lats[(lats < 70) & (lats > 35)]

    fig, axs = plt.subplots(
        3, 3, figsize=(13.5, 9), subplot_kw={"projection": dataproj}
    )
    fig.subplots_adjust(right=0.8)

    vmin_list = [-7, -5, -30]
    vmax_list = [7, 5, 30]
    titles = ["Temperature", "Wind speed", "Radiation"]
    cmaps = [plt.cm.RdBu_r, plt.cm.PiYG, GyYl]
    cbar_units = ["[K]", "[m/s]", "[W/m²]"]
    lon_lim, lat_lim = 33, 70

    overlap_perc = (len(ds_overlapping.time.values) * 100) / (7 * NUM_EVENTS)
    c1_perc = (len(ds_c1.time.values) * 100) / (7 * NUM_EVENTS)
    c2_perc = (len(ds_c2.time.values) * 100) / (7 * NUM_EVENTS)

    psl_overlapping = ds_overlapping.psl.mean(dim="time") / 100  # Pa to hPa
    psl_c1 = ds_c1.psl.mean(dim="time") / 100  # Pa to hPa
    psl_c2 = ds_c2.psl.mean(dim="time") / 100  # Pa to hPa

    tas_overlapping = ds_overlapping.tas.mean(dim="time")
    tas_c1 = ds_c1.tas.mean(dim="time")
    tas_c2 = ds_c2.tas.mean(dim="time")

    sfcWind_overlapping = ds_overlapping.sfcWind.mean(dim="time")
    sfcWind_c1 = ds_c1.sfcWind.mean(dim="time")
    sfcWind_c2 = ds_c2.sfcWind.mean(dim="time")

    rsds_overlapping = ds_overlapping.rsds.mean(dim="time")
    rsds_c1 = ds_c1.rsds.mean(dim="time")
    rsds_c2 = ds_c2.rsds.mean(dim="time")

    titles_column1 = [
        f"Co-occurrence ({overlap_perc:.2f}%)",
        f"{country_combination[0]} unique ({c1_perc:.2f}%)",
        f"{country_combination[1]} unique ({c2_perc:.2f}%)",
    ]
    data_column1 = [
        (psl_overlapping, tas_overlapping, sfcWind_overlapping, rsds_overlapping),
        (psl_c1, tas_c1, sfcWind_c1, rsds_c1),
        (psl_c2, tas_c2, sfcWind_c2, rsds_c2),
    ]

    for j in range(3):
        for i in range(3):
            anom_plot_column1 = data_column1[j][i + 1]
            psl_plot = data_column1[j][0]
            psl_plothl = psl_plot.where(
                (psl_plot.lon > -30)
                & (psl_plot.lon < 33)
                & (psl_plot.lat > 35)
                & (psl_plot.lat < 70),
                drop=True,
            )
            levels_column1 = np.linspace(vmin_list[i], vmax_list[i], 11)
            ax_column1 = axs[i, j]
            ax_column1.set_extent([-30, 33, 35, 70], crs=ccrs.PlateCarree())
            # ax_column1.set_ylim([lats[0], 75])

            norm_column1 = colors.BoundaryNorm(
                levels_column1, ncolors=cmap.N, extend="both"
            )
            im_column1 = ax_column1.contourf(
                lons,
                lats,
                anom_plot_column1,
                transform=ccrs.PlateCarree(),
                levels=levels_column1,
                cmap=cmaps[i],
                norm=norm_column1,
                extend="both",
            )
            fig.colorbar(im_column1, ax=ax_column1, label=cbar_units[i], shrink=0.75)
            CS_column1 = ax_column1.contour(
                lons, lats, psl_plot, transform=ccrs.PlateCarree(), colors="k"
            )
            ax_column1.clabel(CS_column1, inline=True, fontsize=10)
            put.plot_maxmin_points(
                ax_column1,
                lons_plot,
                lats_plot,
                lon_lim,
                lat_lim,
                psl_plothl,
                "max",
                50,
                "H",
                color="k",
                plotValue=False,
                transform=ccrs.PlateCarree(),
            )
            put.plot_maxmin_points(
                ax_column1,
                lons_plot,
                lats_plot,
                lon_lim,
                lat_lim,
                psl_plothl,
                "min",
                50,
                "L",
                color="k",
                plotValue=False,
                transform=ccrs.PlateCarree(),
            )
            ax_column1.coastlines()
            ax_column1.add_feature(cartopy.feature.BORDERS, linestyle=":", alpha=1)
            ax_column1.gridlines(
                crs=ccrs.PlateCarree(),
                draw_labels=False,
                x_inline=False,
                rotate_labels=False,
            )
            # gl.top_labels = False
            # gl.right_labels = False
            # gl.xlocator = ticker.FixedLocator([-20, -10, 0, 10, 20, 30])

            if i == 0:
                ax_column1.set_title(titles_column1[j] + " - " + titles[i])
            else:
                ax_column1.set_title(titles[i])
    plt.tight_layout()
    plt.savefig(
        f"{dir_Figures}/{country_combination}_composite_meteorology.png", dpi=300
    )


# %%
composite_datasets_by_country = create_composite_dataset(df_events)

# %%
# for country in tqdm(df_events.country.unique()):
#     print(f'Starting {country}...')
#     plot_composite_meteorology(country, composite_datasets_by_country)
#     plt.close()

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

country_combinations = []

for key, values in europe_neighbors.items():
    for value in values:
        pair = [key, value]
        reverse_pair = [value, key]
        if (
            pair not in country_combinations
            and reverse_pair not in country_combinations
        ):
            country_combinations.append(pair)

for country_combination in tqdm(country_combinations):
    print(f"Starting {country_combination}...")
    overlapping_datasets, c1_unique_datasets, c2_unique_datasets = (
        meteo_overlapping_countries(country_combination)
    )
    plot_composite_meteorology_countrycomb(
        country_combination,
        overlapping_datasets,
        c1_unique_datasets,
        c2_unique_datasets,
    )
    plt.close()

# %%
