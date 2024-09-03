# %%
import os

import geopandas as gpd
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import xarray as xr
from cartopy.io import shapereader

os.chdir("/usr/people/duinen/MSc-thesis/src/ERA5")
from config_ERA5 import COUNTRIES, PATH_ED
from tqdm import tqdm

# %%
ANOM_PATH = "/net/pc230050/nobackup/users/duinen/LENTIS/present/"
PSL_PATH = "/net/pc200256/nobackup/users/most/LENTIS/present/day/psl_d/"
ERA_PATH = "/net/pc230050/nobackup/users/duinen/LENTIS/ERA5/"
RUNS = np.arange(10, 170)
EXTEND_ERA = [-30, 59, 33, 73]  # lonmin, lonmax, latmin, latmax

# Load the data
ed_lentis = pd.read_csv(
    PATH_ED + "netto_demand_el7_winter_LENTIS_2023_PD_1600_events.csv",
    index_col=0,
)
ed_lentis["runs"] = ed_lentis["runs"].str.slice(1).astype(int)
ed_era5 = pd.read_csv(
    PATH_ED + "demand_net_renewables_netto_demand_el7_winter_ERA5_2023_PD_noHydro_73_events_v2.csv",
    index_col=0,
)


temp_era5 = xr.open_dataset(ERA_PATH + "era5_t2m_d_1950_2022.nc")["t2m"]
wind_era5 = xr.open_dataset(ERA_PATH + "era5_sfcWind_d_1950_2022.nc")["sfcWind"]
clim_era5 = xr.open_dataset(ERA_PATH + "era5_t2m_d_1950_2022_ydaymean.nc")["t2m"]
clim_wind_era5 = xr.open_dataset(ERA_PATH + "era5_sfcWind_d_1950_2022_ydaymean.nc")["sfcWind"]

# temp_era5["time"] = pd.to_datetime(temp_era5["time"].values)
# clim_era5["time"] = pd.to_datetime(clim_era5["time"].values)


def calc_anom_era(data, clim):
    data["dayofyear"] = data["time"].dt.dayofyear
    clim["dayofyear"] = clim["time"].dt.dayofyear

    clim_aligned = clim.groupby("dayofyear").mean("time")

    era5_anom = data.groupby("dayofyear") - clim_aligned
    return era5_anom


def country_mean(lon_country, lat_country, extracted_var):
    return (
        extracted_var.sel(lon=slice(*lon_country), lat=slice(*lat_country))
        .mean(dim=["lon", "lat"])
        .to_numpy()
    )


era5_temp_anom = calc_anom_era(temp_era5, clim_era5)
era5_wind_anom = calc_anom_era(wind_era5, clim_wind_era5)
# %%
# LOAD SHAPEFILE COUNTRIES

resolution = "10m"
category = "cultural"
name = "admin_0_countries"
shpfilename = shapereader.natural_earth(resolution, category, name)
df_shapes = gpd.read_file(shpfilename)


# %%
def compare_lentis_era_anom(country, plot=False):
    global anomaly_lists

    country_bounds = df_shapes.loc[df_shapes["ISO_A3_EH"] == country].geometry.bounds
    lon_country = (country_bounds["minx"].to_numpy()[0], country_bounds["maxx"].to_numpy()[0])
    lat_country = (country_bounds["miny"].to_numpy()[0], country_bounds["maxy"].to_numpy()[0])

    ed_days_era5 = ed_era5[ed_era5["country"] == country]
    ed_days_lentis = ed_lentis[ed_lentis["country"] == country].sort_values("runs")

    temp_event_era5 = []
    wind_event_era5 = []
    temp_event_lentis = []
    wind_event_lentis = []

    for _, row in ed_days_era5.iterrows():
        start_time = row["start_time"]
        end_time = row["end_time"]
        temp_event_era5.append(
            era5_temp_anom.sel(time=slice(start_time, end_time)).mean(dim="time"),
        )
        wind_event_era5.append(
            era5_wind_anom.sel(time=slice(start_time, end_time)).mean(dim="time"),
        )

    for run in tqdm(RUNS):
        run_events_lentis = ed_days_lentis[ed_days_lentis["runs"] == run]
        temp_lentis_run = xr.open_dataset(
            ANOM_PATH + f"tas_d_anomaly/anom_tas_d_ECEarth3_h{run:03d}.nc",
        )["tas"]
        wind_lentis_run = xr.open_dataset(
            ANOM_PATH + f"sfcWind_d_anomaly/anom_sfcWind_d_ECEarth3_h{run:03d}.nc",
        )["sfcWind"]
        for _, row in run_events_lentis.iterrows():
            start_time = row["start_time"]
            end_time = row["end_time"]
            temp_event_lentis.append(
                temp_lentis_run.sel(time=slice(start_time, end_time)).mean(dim="time"),
            )
            wind_event_lentis.append(
                wind_lentis_run.sel(time=slice(start_time, end_time)).mean(dim="time"),
            )
        temp_lentis_run.close()
        wind_lentis_run.close()

    extracted_temp_era5 = xr.concat(temp_event_era5, dim="time")
    extracted_temp_lentis = xr.concat(temp_event_lentis, dim="time")
    extracted_wind_era5 = xr.concat(wind_event_era5, dim="time")
    extracted_wind_lentis = xr.concat(wind_event_lentis, dim="time")

    if plot:
        plt.figure()
        extracted_temp_era5.mean(dim="time").plot()

        plt.figure()
        extracted_temp_lentis.mean(dim="time").plot()

        plt.figure()
        extracted_wind_era5.mean(dim="time").plot()

        plt.figure()
        extracted_wind_lentis.mean(dim="time").plot()

    # TODO: change to actual geometry of country, instead of square
    temp_era5_region = country_mean(lon_country, lat_country, extracted_temp_era5)
    temp_lentis_region = country_mean(lon_country, lat_country, extracted_temp_lentis)
    wind_era5_region = country_mean(lon_country, lat_country, extracted_wind_era5)
    wind_lentis_region = country_mean(lon_country, lat_country, extracted_wind_lentis)

    anomaly_lists[country] = {
        "temp_era": temp_era5_region,
        "temp_lentis": temp_lentis_region,
        "wind_era": wind_era5_region,
        "wind_lentis": wind_lentis_region,
    }

    if plot:
        fig, axs = plt.subplots(1, 2, figsize=(14, 7))

        # Get the line width of the y-axis gridline
        yaxis_line = axs[0].get_yaxis().get_gridlines()[0]
        line_width = yaxis_line.get_linewidth()

        # Temperature anomalies
        sns.kdeplot(temp_era5_region, label="ERA5", ax=axs[0])
        # plt.hist(temp_era5_region, bins=30, alpha=0.5, label="ERA5", density = True, ax=axs[0])
        sns.kdeplot(temp_lentis_region, label="LENTIS", ax=axs[0])
        axs[0].axvline(x=0, color="k", linestyle="-", linewidth=line_width)
        axs[0].axvline(np.mean(temp_era5_region), color="C0", linestyle="--")
        axs[0].axvline(np.mean(temp_lentis_region), color="C1", linestyle="--")
        axs[0].legend()
        axs[0].set_xlabel("Temperature [K]")
        axs[0].set_ylabel("Density")
        axs[0].set_title(f"PDF of event T2m anomalies in {country}")

        # Wind speed anomalies
        sns.kdeplot(wind_era5_region, label="ERA5", ax=axs[1])
        sns.kdeplot(wind_lentis_region, label="LENTIS", ax=axs[1])
        axs[1].axvline(x=0, color="k", linestyle="-", linewidth=line_width)
        axs[1].axvline(np.mean(wind_era5_region), color="C0", linestyle="--")
        axs[1].axvline(np.mean(wind_lentis_region), color="C1", linestyle="--")
        axs[1].legend()
        axs[1].set_xlabel("10m wind speed [m/s]")
        axs[1].set_ylabel("Density")
        axs[1].set_title(f"PDF of event 10m wind speed anomalies in {country}")

        plt.tight_layout()
        plt.show()

    mean_event_temp_era5 = extracted_temp_era5.mean(dim="time").sel(
        lon=slice(EXTEND_ERA[0], EXTEND_ERA[1]),
        lat=slice(EXTEND_ERA[2], EXTEND_ERA[3]),
    )

    mean_event_temp_lentis = extracted_temp_lentis.mean(dim="time").sel(
        lon=slice(EXTEND_ERA[0], EXTEND_ERA[1]),
        lat=slice(EXTEND_ERA[2], EXTEND_ERA[3]),
    )
    mean_event_wind_era5 = extracted_wind_era5.mean(dim="time").sel(
        lon=slice(EXTEND_ERA[0], EXTEND_ERA[1]),
        lat=slice(EXTEND_ERA[2], EXTEND_ERA[3]),
    )
    mean_event_wind_lentis = extracted_wind_lentis.mean(dim="time").sel(
        lon=slice(EXTEND_ERA[0], EXTEND_ERA[1]),
        lat=slice(EXTEND_ERA[2], EXTEND_ERA[3]),
    )

    corr_temp = np.corrcoef(
        mean_event_temp_era5,
        mean_event_temp_lentis,
    )[0, 1]
    corr_wind = np.corrcoef(
        mean_event_wind_era5,
        mean_event_wind_lentis,
    )[0, 1]
    print(f"correlation temp ERA5-LENTIS: {corr_temp:.3f}")
    print(f"correlation wind ERA5-LENTIS: {corr_wind:.3f}")


# %%
anomaly_lists = {}

for country in tqdm(COUNTRIES):
    compare_lentis_era_anom(country)

# %% plot box plots of the anomalies
HOMEDIR = "/usr/people/duinen/MSc-thesis/"
ROW_ORDER = np.load(
    f"{HOMEDIR}Data/row_order_nettodemand_v20240220.npy",
)  # load the row ordering of the clustered residual heatmap, to follow the same clustering!

countries_ordered = np.array(COUNTRIES)[ROW_ORDER]

fig, axs = plt.subplots(2, figsize=(15, 5), sharex=True, dpi=300)

# Prepare data
temp_data_era = [anomaly_lists[country]["temp_era"] for country in countries_ordered]
temp_data_lentis = [anomaly_lists[country]["temp_lentis"] for country in countries_ordered]
wind_data_era = [anomaly_lists[country]["wind_era"] for country in countries_ordered]
wind_data_lentis = [anomaly_lists[country]["wind_lentis"] for country in countries_ordered]

# Create positions for box plots
positions = np.arange(len(COUNTRIES)) * 2
width = 0.4

# Plot data
axs[0].boxplot(
    temp_data_era,
    positions=positions - width / 2,
    widths=width,
    patch_artist=True,
    boxprops=dict(facecolor="C1"),
    medianprops=dict(color="k"),
    showfliers=False,
)
axs[0].boxplot(
    temp_data_lentis,
    positions=positions + width / 2,
    widths=width,
    patch_artist=True,
    boxprops=dict(facecolor="C0"),
    medianprops=dict(color="k"),
    showfliers=False,
)
axs[1].boxplot(
    wind_data_era,
    positions=positions - width / 2,
    widths=width,
    patch_artist=True,
    boxprops=dict(facecolor="C1"),
    medianprops=dict(color="k"),
    showfliers=False,
)
axs[1].boxplot(
    wind_data_lentis,
    positions=positions + width / 2,
    widths=width,
    patch_artist=True,
    boxprops=dict(facecolor="C0"),
    medianprops=dict(color="k"),
    showfliers=False,
)

axs[0].set_ylabel("Temp. anomaly [K]")
axs[1].set_ylabel("Wind anomaly [m/s]")

axs[0].set_xticks([])
axs[0].set_xticklabels([])
axs[1].set_xticks(positions)
axs[1].set_xticklabels(countries_ordered)

era5_patch = mpatches.Patch(facecolor="C1", edgecolor="k", label="ERA5")
lentis_patch = mpatches.Patch(facecolor="C0", edgecolor="k", label="LENTIS")

axs[0].legend(handles=[era5_patch, lentis_patch])

axs[0].axhline(y=0, color="k", linestyle="--", linewidth=0.5)
axs[1].axhline(y=0, color="k", linestyle="--", linewidth=0.5)
plt.show()


# %%
