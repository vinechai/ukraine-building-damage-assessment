# loads the pwtt benchmark: https://github.com/oballinger/PWTT
#
# each city is a csv with one row per building: its shape, a damaged/undamaged label
# (unosat's manually marked damage points joined onto microsoft building footprints),
# and a few sentinel-1 change detection stats pwtt already computed for it.
#
# no earth engine account needed for any of this, it's just files on disk.
# get the data first with scripts/download_benchmark.py

from __future__ import annotations

import json
import re
from pathlib import Path

import geopandas as gpd
import pandas as pd
from shapely.geometry import shape

from .damage_classes import DAMAGED, UNDAMAGED

DEFAULT_ROOT = Path("data/benchmark/one_month")

_FILENAME_RE = re.compile(r"^(?P<city>[A-Za-z]+)_(?P<date>\d{8})_(?P<window>\d+)_footprints\.csv$")

# pwtt's own recommended default threshold on max_change (see their readme).
# note: this is not what they used for their published per-city accuracy table,
# that used a different threshold picked per city, and evaluated at 500m grid
# level instead of per building. this is just a simple baseline rule for us to
# compare our own model against.
DEFAULT_PWTT_THRESHOLD = 3.3


def list_available_cities(root: Path = DEFAULT_ROOT) -> pd.DataFrame:
    """scan the benchmark folder and list which cities are available."""
    rows = []
    for path in sorted(Path(root).glob("*_footprints.csv")):
        m = _FILENAME_RE.match(path.name)
        if not m:
            continue
        rows.append({
            "city": m.group("city"),
            "annotation_date": m.group("date"),
            "path": path,
        })
    return pd.DataFrame(rows)


def load_city(city: str, root: Path = DEFAULT_ROOT) -> gpd.GeoDataFrame:
    """load one city's buildings as a geopandas table, with a `damaged` column
    (renamed from the raw `class` 0/1) plus the sentinel-1 stats pwtt precomputed."""
    catalog = list_available_cities(root)
    matches = catalog[catalog["city"].str.lower() == city.lower()]
    if matches.empty:
        available = ", ".join(sorted(catalog["city"]))
        raise FileNotFoundError(f"no benchmark file for city {city!r}. available: {available}")

    df = pd.read_csv(matches.iloc[0]["path"])
    geometry = df[".geo"].apply(lambda g: shape(json.loads(g)))
    gdf = gpd.GeoDataFrame(df.drop(columns=[".geo"]), geometry=geometry, crs="EPSG:4326")
    gdf["damaged"] = gdf["class"].map({0: UNDAMAGED, 1: DAMAGED})
    gdf["city"] = matches.iloc[0]["city"]
    return gdf


def load_cities(cities: list[str], root: Path = DEFAULT_ROOT) -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(pd.concat([load_city(c, root) for c in cities], ignore_index=True))


def pwtt_baseline_predictions(gdf: gpd.GeoDataFrame, threshold: float = DEFAULT_PWTT_THRESHOLD) -> pd.Series:
    """simple baseline: flag a building damaged if its change score is above threshold."""
    return (gdf["max_change"] > threshold).astype(int)
