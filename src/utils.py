import ast
import glob
import os
from collections import Counter

import numpy as np
import pandas as pd
import xarray as xr
from tqdm import tqdm


def check_make_dir(dir_Name):
    """Create a directory if it does not exist yet.

    Parameters
    ----------
        dir_name (str): Name of the directory to be created.

    Returns
    -------
        None

    """
    if not os.path.exists(dir_Name):
        print("Creating dir %s" % dir_Name)
        os.makedirs(dir_Name)


def extract_country_name(filepath):
    """Extract the country name from a file path.

    Parameters
    ----------
        filepath (str): Path of the file.

    Returns
    -------
        str: Extracted country name.

    """
    return os.path.basename(filepath).split("_")[0]


def add_country_coord(ds, country_name):
    """Assign the country identifier as a coordinate to the dataset.

    Parameters
    ----------
        ds (xarray.Dataset): Input dataset to which country names should be assigned.
        country_name (str): Country name to be assigned.

    Returns
    -------
        xarray.Dataset: Dataset with added country coordinates.

    """
    return ds.assign_coords(country_name=country_name)


def combine_variables(ds, variable):
    """Combine specific variables in a dataset.

    Parameters
    ----------
        ds (xarray.Dataset): Input dataset.
        variable (str): Variable to be combined ('sun', 'wind', or others).

    Returns
    -------
        xarray.DataArray: Dataset with the combined variable.

    """
    if variable == "sun":
        return ds["pv_util"] + ds["pv_roof"]
    elif variable == "wind":
        return ds["wind_offshore"] + ds["wind_onshore"]
    else:
        return ds[variable]


def preprocess_data(filepath, variable):
    """Preprocess data from a file, adding country coordinates and combining variables, if necessary.

    Parameters
    ----------
        filepath (str): Path of the file.
        variable (str): Variable to be combined ('sun', 'wind', or others).

    Returns
    -------
        xarray.Dataset: Preprocessed dataset.

    """
    country_name = extract_country_name(filepath)
    ds = xr.open_dataset(filepath)
    ds = add_country_coord(ds, country_name)
    ds = combine_variables(ds, variable)
    return ds


def load_data(energy_path, variable, runname, stacked=True):
    """Load and preprocess multiple datasets, concatenating them along the 'country' dimension.

    Parameters
    ----------
        energy_path (str): Path to the energy datasets.
        variable (str): Variable to be combined ('sun', 'wind', or others).
        stacked (bool): Whether to stack the dataset along the 'runs, time' dimension.

    Returns
    -------
        xarray.Dataset: Concatenated and preprocessed dataset.

    """
    filepaths = sorted(glob.glob(energy_path + "???" + "_" + runname + ".nc"))
    datasets = [preprocess_data(fp, variable) for fp in filepaths]
    ds = xr.concat(datasets, dim="country")
    ds = ds.where(ds["time"].dt.month.isin([10, 11, 12, 1, 2, 3, 4]), drop=True)
    ds = ds.drop_sel(
        country=[19, 23],
    )  # drop countries that are not properly represented in the analysis

    if stacked:
        return ds.stack(event=("runs", "time"))
    else:
        return ds


def load_anomaly(base_anom, dtype):
    """Load anomaly data for a specific variable type.

    Parameters
    ----------
        base_anom (str): Base directory containing anomaly data files.
        dtype (str): Data type to load ('tas', 'sfcWind', or 'rsds').

    Returns
    -------
        xr.Dataset: Concatenated anomaly data along the 'run' dimension.
        numpy.ndarray: Array of longitudes.
        numpy.ndarray: Array of latitudes.

    """
    files = glob.glob(
        base_anom + f"{dtype}_d_anomaly" + f"/anom_{dtype}_d_ECEarth3_h*.nc",
    )
    files.sort()  # sort the list in place
    datasets = []
    for f in tqdm(files):
        dset = xr.open_dataset(f)
        datasets.append(
            dset[dtype].where(dset["time"].dt.month.isin([11, 12, 1, 2, 3]), drop=True),
        )

    data = xr.concat(datasets, dim="run")
    data["run"] = data["run"] + 10
    return data, data["lon"].values, data["lat"].values


def load_psl(base_psl):
    """Load sea-level pressure (psl) data.

    Parameters
    ----------
        base_psl (str): Base directory containing psl data files.

    Returns
    -------
        xr.Dataset: Concatenated psl data along the 'run' dimension.

    """
    files_psl = glob.glob(base_psl + "psl_d_ECEarth3_h*.nc")
    files_psl.sort()
    datasets_psl = []
    for f in tqdm(files_psl):
        dset = xr.open_dataset(f)
        datasets_psl.append(
            dset.psl.where(dset["time"].dt.month.isin([11, 12, 1, 2, 3]), drop=True),
        )

    data_psl = xr.concat(datasets_psl, dim="run")
    data_psl["run"] = data_psl["run"] + 10
    return data_psl


def merge_cluster_data(cluster_path, data_rank):
    """Merge clustered weather regimes data with ranked energy data based on time and run.

    Parameters
    ----------
        cluster_path (str): Path to the cluster data CSV file.
        data_rank (xr.Dataset): Ranked energy data.

    Returns
    -------
        pd.DataFrame: Merged DataFrame containing energy times series with matching cluster information.

    """
    df_all = pd.read_csv(cluster_path)
    df_all = df_all[["time", "run", "cluster_id"]]
    df_all["time"] = pd.to_datetime(df_all["time"])
    df_all["time"] = df_all["time"].apply(
        lambda dt: dt.replace(hour=12, minute=0, second=0),
    )  # set time to noon, to match df. Is daily average anyway

    df_xr = data_rank.to_dataframe().reset_index(drop=True)
    df_xr["run"] = df_xr["runs"].str.extract(r"(\d+)").astype(int)
    df_xr = df_xr.drop("runs", axis=1)

    df_raw = pd.merge(
        df_xr,
        df_all[["time", "run", "cluster_id"]],
        on=["time", "run"],
        how="left",
    )  # merge both df's on correct time and run
    return df_raw


def df_thresholds(data_stacked, thres):
    """Calculate country-specific energy threshold values from stacked data.

    Parameters
    ----------
        data_stacked (xr.DataArray): Stacked data along the 'event' dimension.
        thres (float): Threshold quantile value.

    Returns
    -------
        pd.DataFrame: DataFrame containing country names and corresponding energy thresholds.

    """
    threshold = data_stacked.quantile(thres, dim="event")
    data = []
    columns = ["Country_name", "threshold"]
    for i, country_name in enumerate(threshold.country_name.values):
        data.append([country_name, float(threshold[i].values)])

    return pd.DataFrame(data, columns=columns)


def calc_composite_mean(data, df_events):
    """Calculate composite mean based on selected events.

    Parameters
    ----------
        data (xr.DataArray): Data array with meteorological data, of which to calculate the composite mean.
        df_events (pd.DataFrame): DataFrame containing time and run information for selected events.

    Returns
    -------
        np.ndarray: Composite mean array.

    """
    num_events = len(df_events["time"])
    composite_mean = (
        np.nansum(
            [
                data.sel(time=t, run=r)
                for t, r in zip(df_events["time"].values, df_events["run"].values)
            ],
            axis=0,
        )
        / num_events
    )
    return composite_mean


def calc_composite_mean_multipledayevent(data_rolling, df_events):
    """Calculate composite mean for multiple day event based on selected events.

    Parameters
    ----------
        data (xr.DataArray): Data array with meteorological data, of which to calculate the composite mean.
        df_events (pd.DataFrame): DataFrame containing time and run information for selected events.
        window (int): Window size for rolling mean.

    Returns
    -------
        np.ndarray: Composite mean array.

    """
    num_events = len(df_events)
    composite_mean = (
        np.nansum(
            [
                data_rolling.sel(time=t, run=r)
                for t, r in zip(df_events["end_time"].values, df_events["run"].values)
            ],
            axis=0,
        )
        / num_events
    )
    return composite_mean


def find_dominant_wr_v2(window_series):
    # counts = window.value_counts()
    counts = window_series.value_counts()
    max_count = counts.max()
    # Check if the maximum count is at least 4
    if max_count >= 4:
        return counts.idxmax()
    # Return 5 if no cluster is dominant
    return 5


# OLD VERSION; MAKE SURE TO NOT USE THIS ONE ANYMORE AND REPLACE BY V2.
def find_dominant_wr(df_events, df_wr, cluster_col, wri=True):
    """Identifies the dominant weather regime for each event in the events DataFrame, based
    on weather regime data. Updates the original events DataFrame with the dominant weather
    regime and the list of weather regime IDs occurring within each event's duration.

    Args:
    ----
        df_events (pd.DataFrame): DataFrame containing events with start and end times,
                                  and a 'run' identifier for each event.
        df_wr (pd.DataFrame): DataFrame containing weather regime data with timestamps, 'run'
                              identifiers, and weather regime IDs specified by `cluster_col`.
        cluster_col (str): Column name in df_wr that contains weather regime IDs.
        wri (bool): Whether to calculate the average and max WRI at the dominant weather regime.

    Returns:
    -------
        pd.DataFrame: The updated events DataFrame including the dominant weather regime for each
                      event and a list of all weather regime IDs occurring within the event's duration.

    Notes:
    -----
        The dominant weather regime is identified as the most common weather regime during the
        event's duration. If no regime occurs at least 4 out of 7 days (assuming a week-long event),
        a value of 5 is assigned to indicate no persisting regime. If only one type of regime is
        present, it is automatically considered dominant.

    """
    # Initialize columns for weather regime IDs and the dominant weather regime
    df_events["weather_regime_ids"] = None
    df_events["dominant_weather_regime"] = None
    df_events["wri_at_dominant_regime"] = None
    df_events["max_wri_at_dominant_regime"] = None

    for index, row in df_events.iterrows():
        run_condition = df_wr["run"] == row["run"]
        date_condition = (df_wr["time"] >= row["start_time"]) & (df_wr["time"] <= row["end_time"])

        # Filter df_wr based on 'run' and date range to find relevant weather regimes
        relevant_weather_regimes = df_wr.loc[
            run_condition & date_condition,
            cluster_col,
        ].tolist()

        if wri:
            relevant_wri = np.array(
                df_wr.loc[
                    run_condition & date_condition,
                    "wri",
                ]
                .apply(ast.literal_eval)
                .to_list(),
            )

            # Add a column of NaNs to the relevant_wri array to avoid shape mismatch
            nan_column = np.full((relevant_wri.shape[0], 1), np.nan)
            relevant_wri = np.hstack((relevant_wri, nan_column))

            wri_at_regimes = [
                relevant_wri[i, relevant_weather_regimes[i]]
                for i in range(len(relevant_weather_regimes))
            ]

        # Identify the top two most common regimes (if applicable) and determine dominance
        dominant_2 = Counter(relevant_weather_regimes).most_common(2)
        if len(dominant_2) == 1 or dominant_2[0][1] >= 4:
            dominant_wr = dominant_2[0][0]
            df_events.loc[index, "dominant_weather_regime"] = dominant_wr
            if wri and dominant_wr != 4:
                indices = np.where(np.array(relevant_weather_regimes) == dominant_wr)[0]
                wri_at_dominant_regime = [wri_at_regimes[i] for i in indices]
                df_events.loc[index, "wri_at_dominant_regime"] = np.mean(wri_at_dominant_regime)
                df_events.loc[index, "max_wri_at_dominant_regime"] = np.max(wri_at_dominant_regime)
        else:
            df_events.loc[index, "dominant_weather_regime"] = 5  # Indicate no persisting regime
            df_events.loc[index, "wri_at_dominant_regime"] = np.nan
            df_events.loc[index, "max_wri_at_dominant_regime"] = np.nan

        # Update the 'weather_regime_ids' column with the list of regimes for each event
        df_events.at[index, "weather_regime_ids"] = relevant_weather_regimes

    return df_events


def get_color(country, regions):
    """Returns the color associated with a given country based on its regional classification dictionary."""
    for region in regions.values():
        if country in region["countries"]:
            return region["color"]
    return None


def reorder(matrix, order):
    df_matrix = pd.DataFrame(matrix)

    matrix_reordered = df_matrix.reindex(order, axis=0)
    matrix_reordered = matrix_reordered.reindex(order, axis=1)
    return matrix_reordered.values
