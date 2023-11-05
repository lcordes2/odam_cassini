# Import libraries
import os

import pandas as pd
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from joblib import dump, load

YEAR_START_DAM = 1975
# Update working directory path as per your setup
wd = "/home/cedric/repos/cassini_data/uk_first_basin/"
# Define the new folder path
data_lars = wd + "data_for_lars/"
# Create the new folder if it doesn't exist
os.makedirs(data_lars, exist_ok=True)

river_downstream_data = pd.read_csv(
    wd + "CAMELS_GB_hydromet_timeseries_31006_19701001-20150930.csv"
)

river_downstream_data["date"] = pd.to_datetime(river_downstream_data["date"])

river_downstream_data_post_dam = river_downstream_data[
    river_downstream_data["date"].dt.year >= YEAR_START_DAM
]


common_days_post_dam_df = river_downstream_data_post_dam[
    [
        "precipitation",
        "pet",
        "temperature",
    ]
]
downstream_flux = river_downstream_data_post_dam[["discharge_vol"]]

# Number of days to look back
number_days_prior = 30  # for one week; adjust this as needed
df_with_lag = common_days_post_dam_df.copy()
# Create lagged features for the number of days you want to look back
for day in range(1, number_days_prior + 1):
    shifted_precipitation = common_days_post_dam_df[["precipitation"]].shift(
        periods=day
    )
    shifted_precipitation.columns = [
        str(col) + f"_lag_{day}" for col in shifted_precipitation.columns
    ]
    df_with_lag = pd.concat([df_with_lag, shifted_precipitation], axis=1)

df_with_lag = df_with_lag.dropna()
# Load the trained model
best_model = load(wd + "qi_estimator.joblib")

# Load the scaler
scaler = load(wd + "scaler.joblib")
X = scaler.fit_transform(df_with_lag)


# Perform predictions
upstream_flux = best_model.predict(X)

downstream_flux = downstream_flux[number_days_prior:]
water_stored = upstream_flux - downstream_flux["discharge_vol"]
water_stored = water_stored.clip(0)

# Convert water_stored to a DataFrame
water_stored_df = pd.DataFrame({"water_stored": water_stored})

# Set the date as the index
water_stored_df['date'] = river_downstream_data_post_dam["date"][
    number_days_prior:
]

# Resample to get the average per month
monthly_average_water_stored = water_stored_df.resample("M", on="date").sum()

# Reset the index to turn 'date' back into a column
monthly_average_water_stored.reset_index(inplace=True)


# Plotting
plt.figure(figsize=(15, 5))
plt.plot(
    monthly_average_water_stored["date"].to_list(),
    monthly_average_water_stored["water_stored"],
    label="Water stored",
    color="blue",
)
plt.title("Water stored over time")
plt.xlabel("Time")
plt.ylabel("Water Volume")
plt.legend()
plt.show()

# Save the results to a CSV file
common_days_post_dam_df.to_csv(
    wd + "post_dam_predictions_with_discharge_vol.csv", index=False
)


# group by year
# Set the date as the index
# Ensure 'date' is a datetime column
water_stored_df['date'] = pd.to_datetime(water_stored_df['date'])

# Add a 'month' column for grouping
monthly_average_water_stored['month'] = monthly_average_water_stored['date'].dt.month

monthly_average_all_years = monthly_average_water_stored.groupby('month')['water_stored'].mean().reset_index()

# Plotting
plt.figure(figsize=(15, 5))
plt.plot(
    monthly_average_all_years["month"].to_list(),
    monthly_average_all_years["water_stored"],
    label="Water stored",
    color="blue",
)
plt.title("Water stored over time")
plt.xlabel("Time")
plt.ylabel("Water Volume")
plt.legend()
plt.show()

## save some data
# Define the path for the new CSV file
csv_file_path = os.path.join(data_lars, "monthly_average_water_stored.csv")

# Save the DataFrame to CSV
monthly_average_all_years.to_csv(csv_file_path, index=False)