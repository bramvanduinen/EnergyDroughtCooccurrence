# config.py

# Constants
DATA_SOURCE = "LENTIS"  # "LENTIS" or "ERA5"
if DATA_SOURCE == "ERA5":
    FOLDER = "/net/pc200256//nobackup/users/most/output/ERA5_2023_PD/agg_production/per_country/"
    RUNNAME = "ERA5_2023_PD_noHydro"
elif DATA_SOURCE == "LENTIS":
    FOLDER = "/net/pc200256//nobackup/users/most/output/LENTIS_2023_PD/agg_production/per_country/"
    RUNNAME = "LENTIS_2023_PD"  # without dispatch hydro
    # RUNNAME = "LENTIS_2023_PD_pf2"  # with dispatch hydro

OFOLDER = "/usr/people/duinen/MSc-thesis/src/find_energydroughts/data/"
EVENTTYPE = "residual_load"
COUNTRIES = [
    "AUT",
    "BEL",
    "CHE",
    "CZE",
    "DEU",
    "DNK",
    "ESP",
    "EST",
    "FIN",
    "FRA",
    "GBR",
    "HRV",
    "HUN",
    "IRL",
    "ITA",
    "LTU",
    "LVA",
    "NLD",
    "NOR",
    "POL",
    "PRT",
    "SVK",
    "SVN",
    "SWE",
]
