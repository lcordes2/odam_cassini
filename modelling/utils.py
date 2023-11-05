from json import load

import ee
import pandas as pd

from ee.geometry import Geometry
from ee.imagecollection import ImageCollection
from pandas import DataFrame

from joblib import load

# Load the scaler from the training script
scaler = load('/home/cedric/repos/cassini_data/uk_first_basin/' + "scaler.joblib")
NUMBER_DAYS_PRIOR = 30
# Function to preprocess data similarly to the training script

def get_spatial_mean(
    image: ImageCollection,
    polygon: Geometry,
    band: str,
) -> ImageCollection:
    # we set best effort to false to prevent issue with max number of pixels
    mean = image.reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=polygon,
        scale=1000,
        bestEffort=False,
        maxPixels=10000000000,
    ).get(band)
    return image.set("daily_mean_temp", mean)


def preprocess_data(df:DataFrame) -> DataFrame:


    # Create lagged features for precipitation
    df_with_lag = df.copy()
    for day in range(1, NUMBER_DAYS_PRIOR + 1):
        shifted_precipitation = df[["precipitation"]].shift(periods=day)
        shifted_precipitation.columns = [
            f"{col}_lag_{day}" for col in shifted_precipitation.columns
        ]
        df_with_lag = pd.concat([df_with_lag, shifted_precipitation], axis=1)

    # Drop the initial period that doesn't have enough data to form a lag
    df_with_lag = df_with_lag[NUMBER_DAYS_PRIOR:]

    # Scale features
    df_with_lag_scaled = scaler.transform(df_with_lag)

    return pd.DataFrame(
        df_with_lag_scaled, index=df_with_lag.index, columns=df_with_lag.columns
    )