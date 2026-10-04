# pulls the raw sentinel-1 signal per building over time, straight from google earth engine.
# this is the part pwtt's own csvs don't have: they only kept one squashed summary number
# per building (max_change etc), not the actual time series behind it.
#
# needs a free earth engine account + a registered cloud project, see readme.
#
# how it works: sentinel-1 revisits the same spot roughly every 12 days. each visit gives
# two radar values per building (VV and VH, two different polarizations). we grab those two
# numbers for every visit in a time window, for every building. that's the "signal" a model
# can actually learn from, instead of pwtt's single pre-computed change score.
#
# one thing that matters: sentinel-1 looks at the ground from a specific angle depending on
# which orbit it's on, and different orbits give slightly different looking images of the
# same spot. mixing orbits together would look like "change" even with nothing happening on
# the ground. so we pick one orbit (whichever one covered the area most often) and stick to it.

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed

import ee
import geopandas as gpd
import pandas as pd

S1_COLLECTION = "COPERNICUS/S1_GRD"


def _s1_base_collection(aoi: ee.Geometry, start: str, end: str) -> ee.ImageCollection:
    return (
        ee.ImageCollection(S1_COLLECTION)
        .filterBounds(aoi)
        .filterDate(start, end)
        .filter(ee.Filter.eq("instrumentMode", "IW"))
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH"))
    )


def pick_best_orbit(aoi: ee.Geometry, start: str, end: str) -> tuple[int, int]:
    """find which orbit track covered this area the most times in the window.
    returns (orbit_number, image_count)."""
    base = _s1_base_collection(aoi, start, end)
    orbit_counts = (
        base.reduceColumns(ee.Reducer.frequencyHistogram(), ["relativeOrbitNumber_start"])
        .get("histogram")
        .getInfo()
    )
    if not orbit_counts:
        raise ValueError("no sentinel-1 images found for this area/time window")
    best_orbit = max(orbit_counts, key=orbit_counts.get)
    return int(float(best_orbit)), int(orbit_counts[best_orbit])


def _buildings_to_fc(gdf: gpd.GeoDataFrame, id_col: str = "FID") -> ee.FeatureCollection:
    features = []
    for _, row in gdf.iterrows():
        geom = ee.Geometry(row.geometry.__geo_interface__)
        features.append(ee.Feature(geom, {"building_id": int(row[id_col])}))
    return ee.FeatureCollection(features)


# earth engine refuses a synchronous request once it would return more than 5000
# elements. one image x one city can already blow past that (a city can have over
# 100k buildings), so we split the buildings into chunks and pull them separately.
CHUNK_SIZE = 4000


def _buildings_to_fc_chunks(gdf: gpd.GeoDataFrame, id_col: str = "FID") -> list[ee.FeatureCollection]:
    chunks = []
    for start in range(0, len(gdf), CHUNK_SIZE):
        chunks.append(_buildings_to_fc(gdf.iloc[start:start + CHUNK_SIZE], id_col=id_col))
    return chunks


def extract_time_series(
    gdf: gpd.GeoDataFrame,
    war_start: str,
    pre_months: int = 12,
    post_months: int = 1,
    inference_start: str | None = None,
    id_col: str = "FID",
    scale: int = 10,
    orbit: int | None = None,
) -> pd.DataFrame:
    """for every building in gdf, pull VV/VH backscatter for every sentinel-1 pass
    from `pre_months` before war_start up to `post_months` after inference_start
    (inference_start defaults to war_start if not given).

    pass `orbit` explicitly when pulling a second, separate window for a city that
    already has a baseline pulled (e.g. a fresh "current" snapshot). otherwise this
    picks whichever orbit happened to cover this specific window the most, which can
    differ from whatever orbit the original baseline used. mixing orbits between two
    periods you're comparing is a real bug, not a minor inconsistency: different
    orbits look at the ground from different angles, which shows up as apparent
    "change" even when nothing happened, and in the worst case (seen on mariupol) an
    orbit can have plenty of images but almost no valid pixels over the actual area,
    silently producing mostly-NaN readings.

    returns a long table: building_id, date, orbit, VV, VH (one row per building per pass).
    """
    inference_start = inference_start or war_start
    start = ee.Date(war_start).advance(-pre_months, "month")
    end = ee.Date(inference_start).advance(post_months, "month")

    aoi = ee.Geometry(gdf.union_all().envelope.__geo_interface__)
    if orbit is None:
        orbit, n_images = pick_best_orbit(aoi, start.format().getInfo(), end.format().getInfo())
        print(f"using orbit {orbit} ({n_images} images in this window)")
    else:
        print(f"using orbit {orbit} (forced, to match an existing baseline)")

    collection = _s1_base_collection(aoi, start.format().getInfo(), end.format().getInfo()) \
        .filter(ee.Filter.eq("relativeOrbitNumber_start", orbit)) \
        .select(["VV", "VH"])

    image_list = collection.toList(collection.size())
    n_images = image_list.size().getInfo()
    # grab every date in one call instead of one getInfo() per image
    epoch_millis = collection.aggregate_array("system:time_start").getInfo()
    dates = [pd.Timestamp(ms, unit="ms").strftime("%Y-%m-%d") for ms in epoch_millis]

    fc_chunks = _buildings_to_fc_chunks(gdf, id_col=id_col)
    tasks = [(i, c) for i in range(n_images) for c in range(len(fc_chunks))]
    print(f"{n_images} images x {len(fc_chunks)} building batch(es) = {len(tasks)} requests, running in parallel")

    def run_one(task):
        i, c = task
        image = ee.Image(image_list.get(i))
        reduced = image.reduceRegions(collection=fc_chunks[c], reducer=ee.Reducer.mean(), scale=scale)
        features = reduced.getInfo()["features"]
        for f in features:
            f["properties"]["date"] = dates[i]
            f["properties"]["orbit"] = orbit
        return [f["properties"] for f in features]

    rows = []
    done = 0
    with ThreadPoolExecutor(max_workers=20) as pool:
        futures = [pool.submit(run_one, t) for t in tasks]
        for future in as_completed(futures):
            rows.extend(future.result())
            done += 1
            if done % 20 == 0 or done == len(tasks):
                print(f"  {done}/{len(tasks)} requests done")

    return pd.DataFrame(rows)
