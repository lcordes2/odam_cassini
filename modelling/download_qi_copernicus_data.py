import ee
import geopandas as gpd
import pandas as pd
from shapely.ops import unary_union
from pathlib import Path
from utils import get_spatial_mean
from pandas import DataFrame

MODEL_NASA = "CNRM-ESM2-1"


def generate_time_series(
    image_collection_id: str,
    band: str,
    basin_geo_file: Path,
    start_date: str,
    end_date: str,
) -> DataFrame:
    # Initialize the Earth Engine API.
    ee.Initialize()

    basin_gdf = gpd.read_file(basin_geo_file)
    basin_polygon = basin_gdf.geometry.to_list()[0]
    merged_polygon = unary_union(basin_polygon)
    coords_water_basin = list(merged_polygon.exterior.coords)

    ee_water_basin_polygon = ee.Geometry.Polygon(coords_water_basin)

    image_collection = ee.ImageCollection(image_collection_id)

    if image_collection_id == "NASA/GDDP-CMIP6":
        filtered_collection = (
            image_collection.filterBounds(ee_water_basin_polygon).filterDate(
                start_date, end_date
            )
        ).filter(ee.Filter.eq("model", MODEL_NASA))
    else:
        filtered_collection = image_collection.filterBounds(
            ee_water_basin_polygon
        ).filterDate(start_date, end_date)

    daily_mean = filtered_collection.map(
        lambda image: get_spatial_mean(image, ee_water_basin_polygon, band)
    )

    # Reduce the collections to lists
    time_series = (
        daily_mean.reduceColumns(
            ee.Reducer.toList(2), ["system:time_start", "daily_mean_temp"]
        )
        .values()
        .get(0)
    )

    # Get the results as a Python list.
    values = time_series.getInfo()

    df_time_series = pd.DataFrame(values, columns=["timestamp", "mean_daily"])
    df_time_series["timestamp"] = pd.to_datetime(df_time_series["timestamp"], unit="ms")
    df_time_series.set_index("timestamp", inplace=True)

    return df_time_series


path_water_basin = Path(
    "/home/cedric/repos/cassini_data/uk_first_basin/uk_first_basin.gpkg"
)

start_date = "2022-9-05"
end_date = "2022-11-19"

precipitation_time_series = generate_time_series(
    "NASA/GDDP-CMIP6",
    "pr",
    path_water_basin,
    start_date,
    end_date,
)

temperature_time_series = generate_time_series(
    "NASA/GDDP-CMIP6",
    "tas",
    path_water_basin,
    start_date,
    end_date,
)


evapotranspiration_time_series = generate_time_series(
    "ECMWF/ERA5_LAND/DAILY_AGGR",
    "potential_evaporation_sum",
    path_water_basin,
    start_date,
    end_date,
)


# Rename the columns to appropriate names
precipitation_time_series.rename(columns={"mean_daily": "Precipitation"}, inplace=True)
temperature_time_series.rename(columns={"mean_daily": "Temperature"}, inplace=True)
evapotranspiration_time_series.rename(
    columns={"mean_daily": "Evapotranspiration"}, inplace=True
)


# remove possible duplicates
precipitation_time_series = precipitation_time_series.groupby(
    temperature_time_series.index
).mean()
temperature_time_series = temperature_time_series.groupby(
    temperature_time_series.index
).mean()
evapotranspiration_time_series = evapotranspiration_time_series.groupby(
    temperature_time_series.index
).mean()

# Merge the dataframes into a single dataframe
merged_df = pd.concat(
    [
        precipitation_time_series,
        temperature_time_series,
        evapotranspiration_time_series,
    ],
    axis=1,
)

# If the time series indices are not perfectly aligned (meaning the timestamps exactly match),
# you will end up with NaNs in places where the indices do not match.
# You may choose to fill or drop NaNs depending on your requirements.

# Save the merged dataframe as a CSV file
output_file = f"/home/cedric/repos/cassini_data/uk_first_basin/input_data_time_series_{start_date}_{end_date}.csv"
merged_df.to_csv(output_file)
