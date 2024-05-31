# Paths
PATH_CLUSTERS = "/usr/people/duinen/MSc-thesis/Results/clusters/"
PATH_DATA = "/usr/people/duinen/MSc-thesis/Results/data/"
PATH_ED = "/usr/people/duinen/MSc-thesis/src/find_energydroughts/data/"
PATH_ANOM = "/net/pc230050/nobackup/users/duinen/LENTIS/present/"
PATH_ZG500 = "/net/pc230050/nobackup/users/duinen/LENTIS/present/zg500_d/"
PATH_ZG500_ERA5 = "/net/pc230050/nobackup/users/duinen/LENTIS/ERA5/"
PATH_PSL = "/net/pc200256/nobackup/users/most/LENTIS/present/day/psl_d"

# constants
CLUSTER_NAMES = ["NAO +", "NAO -", "Blocking", "Atl. Ridge"]
# CLUSTER_NAMES = ["Blocking", "NAO -", "NAO +", "Atl. Ridge"] # changed 2024-03-15 to match new reordering script
WINDOW = 7

# clustering parameters
VERSION = "Bayes_full_v2"  # update this when you change the clustering algorithm and want to save the new results
USE_SCALING = 0
N_EOFS = 20
N_EOFS_FOR_KMEANS = 20
N_CLUSTERS = 4
THRES = 0.8  # If posterior probability of best-regime is below this threshold, assign to no regime

# energy drought parameters
EVENTTYPE = "random"
EVENT_LENGTH = 7
NUM_EVENTS = 1600
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
# Lieke
FOLDER = "/net/pc200256//nobackup/users/most/output/LENTIS_2023_PD/agg_production/per_country/"
RUNNAME = "LENTIS_2023_PD"
OFOLDER = "data/"
