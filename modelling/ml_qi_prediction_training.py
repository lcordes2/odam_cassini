# import libraries
import os

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from joblib import dump, load
from sklearn.preprocessing import StandardScaler


YEAR_START_DAM = 1975
# Update working directory path as per your setup
wd = "/home/cedric/repos/cassini_data/uk_first_basin/"
data_folder = "data_fedor"  # New folder for storing data

# Create the data_fedor directory if it does not exist
data_fedor_path = os.path.join(wd, data_folder)
if not os.path.exists(data_fedor_path):
    os.makedirs(data_fedor_path)

# Read the data
# merged_df = pd.read_csv(wd + "/merged_input_data_time_series_1970-2020.csv")
river_downstream_data = pd.read_csv(
    wd + "/CAMELS_GB_hydromet_timeseries_31006_19701001-20150930.csv"
)

# Convert timestamps to datetime
# merged_df['timestamp'] = pd.to_datetime(merged_df['timestamp'])
river_downstream_data["date"] = pd.to_datetime(river_downstream_data["date"])

# Filter data to include only dates before the dam was constructed
# merged_df = merged_df[merged_df['timestamp'].dt.year < YEAR_START_DAM]
river_downstream_data = river_downstream_data[
    river_downstream_data["date"].dt.year < YEAR_START_DAM
]

# Perform an inner merge with only the 'discharge_vol' column from the river_downstream_data
# common_days_df = pd.merge(
#     merged_df,
#     river_downstream_data[['date', 'discharge_vol']],
#     left_on='timestamp',
#     right_on='date',
#     how='inner'
# )

common_days_df = river_downstream_data[
    [
        "date",
        "discharge_vol",
        "precipitation",
        "pet",
        "temperature",
    ]
]

# Remove rows with NaN values created by shifting
common_days_df = common_days_df.dropna()

target_data = common_days_df["discharge_vol"]

# common_days_df = common_days_df.drop(
#     ["timestamp", 'discharge_vol', 'date'], axis=1
# )  # Dropping the date column as instructed

common_days_df = common_days_df.drop(
    ["date", "discharge_vol"], axis=1
)  # Dropping the date column as instructed

# Number of days to look back
number_days_prior = 30  # for one week; adjust this as needed
df_with_lag = common_days_df.copy()
# Create lagged features for the number of days you want to look back
for day in range(1, number_days_prior + 1):
    shifted_precipitation = common_days_df[["precipitation"]].shift(periods=day)
    shifted_precipitation.columns = [
        str(col) + f"_lag_{day}" for col in shifted_precipitation.columns
    ]
    df_with_lag = pd.concat([df_with_lag, shifted_precipitation], axis=1)

# we need to drop the first 7 days

common_days_df = df_with_lag[number_days_prior:]
target_data = target_data[number_days_prior:]

# Initialize the StandardScaler
scaler = StandardScaler()

# Fit the scaler on the data and transform it
X_scaled = scaler.fit_transform(common_days_df)

# After fitting the scaler with your training data:
dump(scaler, wd + "scaler.joblib")

# Convert the array back to a dataframe
common_days_df = pd.DataFrame(
    X_scaled, index=common_days_df.index, columns=common_days_df.columns
)


X = common_days_df
y = target_data

# Instead of a percentage-based split, use an index that respects time ordering.
# Determine the split index
split_index = int(len(X) * 0.75)  # Assuming 75% of data for training
dates = river_downstream_data['date'][number_days_prior:]  # Extracting the dates after dropping NaNs

# Split the data by index to maintain the time series order
X_train = X[:split_index]
y_train = y[:split_index]
dates_train = dates[:split_index]
X_test = X[split_index:]
y_test = y[split_index:]
dates_test = dates[split_index:]


#%% Model training ############################################################
# Set hyperparameter space for tuning
hyperparameters = {
    "n_estimators": [400, 500],
    "max_features": [0.8, 0.6, 0.5, 0.3],
}

# Perform Grid Search CV to find the best parameters
rf_cv = GridSearchCV(
    RandomForestRegressor(),
    hyperparameters,
    cv=5,
    return_train_score=True,
    verbose=True,
)
rf_cv.fit(X_train, y_train)
print("Best hyperparameters:", rf_cv.best_params_)
print("Best score:", rf_cv.best_score_)

# Make predictions on the test set
y_pred = rf_cv.best_estimator_.predict(X_test)

# Evaluate the model
rmse = mean_squared_error(y_test, y_pred, squared=False)
r2 = r2_score(y_test, y_pred)
print(f"Root Mean Squared Error: {rmse}")
print(f"R-squared: {r2}")

# Feature importance
fi = pd.DataFrame(
    data=rf_cv.best_estimator_.feature_importances_,
    index=X_train.columns,
    columns=["Importance"],
).sort_values(by="Importance", ascending=False)
print(fi)

# Save the predicted data with dates for the test set
predictions_df = pd.DataFrame({'Date': dates_test.reset_index(drop=True), 'Predicted': y_pred})
predictions_df.to_csv(os.path.join(data_fedor_path, "predictions.csv"), index=False)
# Make predictions on the training set
y_train_pred = rf_cv.best_estimator_.predict(X_train)
# Combine the training dates with the training predictions
training_predictions_df = pd.DataFrame({'Date': dates_train.reset_index(drop=True), 'Predicted': y_train_pred})
training_predictions_df.to_csv(os.path.join(data_fedor_path, "training_predictions.csv"), index=False)

# After training and evaluation:
# Save the model
best_model = rf_cv.best_estimator_
dump(best_model, wd + "qi_estimator.joblib")

# Extract the dates for the test period
dates_test_period = river_downstream_data["date"][split_index:]

# Plotting
plt.figure(figsize=(15, 5))
plt.plot(
    dates_test_period.to_list()[number_days_prior:],
    y_test.to_list(),
    label="Observed",
    color="blue",
)
plt.plot(
    dates_test_period.to_list()[number_days_prior:],
    list(y_pred),
    label="Predicted",
    color="red",
    alpha=0.7,
)
plt.title("Observed vs Predicted Water Volume")
plt.xlabel("Time")
plt.ylabel("Water Volume")
plt.legend()
plt.show()
