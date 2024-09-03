import os

import numpy as np
import pandas as pd
import xarray as xr
from config_edw import EVENTTYPE, FOLDER, OFOLDER, REGIONS, RUNNAME
from tqdm import tqdm

# Constants
RUNS = [f"h{number:03d}" for number in range(10, 170)]
VAR0 = "netto_demand"
NR_OF_EVENTS = 1600  # once a year events
SEASONS = ["winter"]
EVENT_LENGTHS = [7]


def open_energy_dataset(country, folder):
    """Opens the energy dataset for a given country."""
    file_path = os.path.join(folder, f"{country}_{RUNNAME}.nc")
    return xr.open_dataset(file_path)


def select_season_data(dataset, season):
    """Filters dataset based on the specified season."""
    season_months = {
        "summer": [6, 7, 8, 9],
        "winter": [1, 2, 3, 11, 12],
    }
    if season in season_months:
        return dataset.where(dataset.time.dt.month.isin(season_months[season]), drop=True)
    return dataset


def process_events(var_data, variable, event_length, num_events):
    """Processes and identifies energy events in the dataset."""
    events = []
    timestamps = []

    # if variable == "residual_pvwind":
    #     dataset[variable] = (
    #         dataset["demand"]
    #         - dataset["pv_util"]
    #         - dataset["wind_offshore"]
    #         - dataset["wind_onshore"]
    #     )
    # elif variable == "netto_demand":
    #     dataset[variable] = (
    #         dataset["demand"]
    #         - dataset["pv_util"]
    #         - dataset["wind_offshore"]
    #         - dataset["wind_onshore"]
    #         - dataset["ror"]
    #     )

    # var_data = dataset[variable]

    for _ in tqdm(range(num_events)):
        if EVENTTYPE == "random":
            valid_section_found = False
            while not valid_section_found:
                try:
                    random_section_start = np.random.randint(
                        0,
                        len(var_data.time) - event_length + 1,
                    )
                    random_run = np.random.randint(0, 160)
                    random_section = var_data.isel(
                        time=slice(random_section_start, random_section_start + event_length),
                        runs=random_run,
                    ).dropna(dim="time")
                    # Attempt to access the last time value to check for potential IndexError
                    max_timestamp = random_section.time.values[-1]
                    valid_section_found = (
                        True  # If the above line succeeds, mark the section as valid
                    )
                except IndexError:
                    # If an IndexError is caught, the loop will continue until a valid section is found
                    continue
            max_event = random_section.sum()
            max_run = var_data.runs[random_run].values
        else:
            rolling_sum = var_data.rolling(time=event_length).sum()
            max_event = rolling_sum.where(rolling_sum == rolling_sum.max(), drop=True)
            max_run = max_event.runs.values[0]
            max_timestamp = max_event.time.values[0]

        min_timestamp = max_timestamp - np.timedelta64(event_length - 1, "D")

        timeslice = slice(min_timestamp, max_timestamp)
        events.append((max_run, timeslice))
        timestamps.append((max_run, min_timestamp, max_timestamp, max_event.to_numpy().item()))

        event_data = var_data.sel(time=timeslice, runs=max_run).expand_dims("runs")
        event_data.name = "residual_event"

        merged_data = xr.merge([var_data, event_data])
        var_data = var_data.where((merged_data[variable] - merged_data["residual_event"]) != 0)

    return events, timestamps


def main():
    """Process energy datasets."""
    print("Event type:", EVENTTYPE)
    for season in SEASONS:
        for event_length in EVENT_LENGTHS:
            data_frames = []
            for region_key in REGIONS:
                print(f"Processing {NR_OF_EVENTS} {event_length}-day events for {region_key}")
                data_arrays_region = []
                for country in REGIONS[region_key]["countries"]:
                    dataset = open_energy_dataset(country, FOLDER)
                    if VAR0 == "residual_pvwind":
                        dataset[VAR0] = (
                            dataset["demand"]
                            - dataset["pv_util"]
                            - dataset["wind_offshore"]
                            - dataset["wind_onshore"]
                        )
                    elif VAR0 == "netto_demand":
                        dataset[VAR0] = (
                            dataset["demand"]
                            - dataset["pv_util"]
                            - dataset["wind_offshore"]
                            - dataset["wind_onshore"]
                            - dataset["ror"]
                        )

                    season_data_country = select_season_data(dataset, season)
                    var_data = season_data_country[VAR0]
                    data_arrays_region.append(var_data)
                season_data = xr.concat(data_arrays_region, dim="country").sum(dim="country")
                events, timestamps = process_events(
                    season_data,
                    VAR0,
                    event_length,
                    NR_OF_EVENTS,
                )

                df = pd.DataFrame(timestamps, columns=["runs", "start_time", "end_time", VAR0])
                df["country"] = region_key
                data_frames.append(df)

            all_events_df = pd.concat(data_frames).reset_index()
            all_events_df = all_events_df.rename({"index": "event_number"}, axis=1)
            all_events_df["event_number"] += 1

            filename = f"{EVENTTYPE}_regions_{VAR0}_el{event_length}_{season}_{RUNNAME}_{NR_OF_EVENTS}_events_with_quantity.csv"
            output_file = os.path.join(OFOLDER, filename)
            all_events_df.to_csv(output_file)
            print(f"Saved to {output_file}")


if __name__ == "__main__":
    main()
