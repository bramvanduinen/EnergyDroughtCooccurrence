import geopandas
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import xarray as xr
from cartopy.io import shapereader
from config_ERA5 import PATH_ED
from tqdm import tqdm

# %%
LENTIS_PATH = "/net/pc200256/nobackup/users/most/LENTIS/present/day/"
PSL_PATH = "/net/pc200256/nobackup/users/most/LENTIS/present/day/psl_d/"
ERA_PATH = "/net/pc230050/nobackup/users/duinen/LENTIS/ERA5/"
RUNS = np.arange(10, 170)

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


# def calc_anom_era(data, clim):
#     data["dayofyear"] = data["time"].dt.dayofyear
#     clim["dayofyear"] = clim["time"].dt.dayofyear

#     clim_aligned = clim.groupby("dayofyear").mean("time")

#     era5_anom = data.groupby("dayofyear") - clim_aligned
#     return era5_anom


def country_mean(lon_country, lat_country, extracted_var):
    return (
        extracted_var.sel(lon=slice(*lon_country), lat=slice(*lat_country))
        .mean(dim=["lon", "lat"])
        .to_numpy()
    )


# era5_temp_anom = calc_anom_era(temp_era5, clim_era5)
# era5_wind_anom = calc_anom_era(wind_era5, clim_wind_era5)
# %%
# LOAD SHAPEFILE COUNTRIES

resolution = "10m"
category = "cultural"
name = "admin_0_countries"
shpfilename = shapereader.natural_earth(resolution, category, name)
df_shapes = geopandas.read_file(shpfilename)


# %%
def compare_lentis_era_meteo(country):
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
            temp_era5.sel(time=slice(start_time, end_time)).mean(dim="time"),
        )
        wind_event_era5.append(
            wind_era5.sel(time=slice(start_time, end_time)).mean(dim="time"),
        )

    for run in tqdm(RUNS):
        run_events_lentis = ed_days_lentis[ed_days_lentis["runs"] == run]
        temp_lentis_run = xr.open_dataset(
            LENTIS_PATH + f"tas_d/tas_d_ECEarth3_h{run:03d}.nc",
        )["tas"]
        wind_lentis_run = xr.open_dataset(
            LENTIS_PATH + f"sfcWind_d/sfcWind_d_ECEarth3_h{run:03d}.nc",
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

    plt.figure()
    extracted_temp_era5 = xr.concat(temp_event_era5, dim="time")
    extracted_temp_era5.mean(dim="time").plot()

    plt.figure()
    extracted_temp_lentis = xr.concat(temp_event_lentis, dim="time")
    extracted_temp_lentis.mean(dim="time").plot()

    plt.figure()
    extracted_wind_era5 = xr.concat(wind_event_era5, dim="time")
    extracted_wind_era5.mean(dim="time").plot()

    plt.figure()
    extracted_wind_lentis = xr.concat(wind_event_lentis, dim="time")
    extracted_wind_lentis.mean(dim="time").plot()

    # TODO: change to actual geometry of country, instead of square
    temp_era5_region = country_mean(lon_country, lat_country, extracted_temp_era5)
    temp_lentis_region = country_mean(lon_country, lat_country, extracted_temp_lentis)
    wind_era5_region = country_mean(lon_country, lat_country, extracted_wind_era5)
    wind_lentis_region = country_mean(lon_country, lat_country, extracted_wind_lentis)

    plt.figure()
    sns.kdeplot(temp_era5_region, label="ERA5")
    sns.kdeplot(temp_lentis_region, label="LENTIS")
    plt.axvline(np.mean(temp_era5_region), color="C0", linestyle="--")
    plt.axvline(np.mean(temp_lentis_region), color="C1", linestyle="--")

    plt.legend()
    plt.xlabel("Temperature")
    plt.ylabel("Density")
    plt.title(f"PDF of event T2m in {country}")

    plt.show()

    plt.figure()
    sns.kdeplot(wind_era5_region, label="ERA5")
    sns.kdeplot(wind_lentis_region, label="LENTIS")
    plt.axvline(np.mean(wind_era5_region), color="C0", linestyle="--")
    plt.axvline(np.mean(wind_lentis_region), color="C1", linestyle="--")

    plt.legend()
    plt.xlabel("10m wind speed (m/s)")
    plt.ylabel("Density")
    plt.title(f"PDF of event 10m wind speed in {country}")

    plt.show()
    # return temp_era5_region


compare_lentis_era_meteo("NLD")
# %%
country_list = ["DEU", "IRL", "NOR", "ITA", "POL", "EST", "ESP", "NLD"]
for country in country_list:
    compare_lentis_era_meteo(country)

# %%
