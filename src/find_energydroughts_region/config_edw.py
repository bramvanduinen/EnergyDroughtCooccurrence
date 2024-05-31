# config.py

# Constants
FOLDER = "/net/pc200256//nobackup/users/most/output/LENTIS_2023_PD/agg_production/per_country/"
RUNNAME = "LENTIS_2023_PD"
OFOLDER = "/usr/people/duinen/MSc-thesis/src/find_energydroughts_region/data/"
EVENTTYPE = "max_drought"
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

REGIONS = {
    "IB": {"color": "C4", "countries": ["PRT", "ESP"]},
    "NW": {"color": "C1", "countries": ["NLD", "DEU", "BEL", "DNK", "GBR", "IRL"]},
    "N": {"color": "C0", "countries": ["NOR", "SWE", "FIN"]},
    "B": {"color": "cyan", "countries": ["EST", "LVA", "LTU"]},
    "C": {"color": "C2", "countries": ["AUT", "ITA", "CHE", "FRA"]},
    "E": {"color": "C3", "countries": ["POL", "HRV", "HUN", "SVN", "CZE", "SVK"]},
}
