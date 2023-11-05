import numpy as np
import pandas as pd
import os
from joblib import load

# Assuming preprocess_data is defined in utils.py and it works as intended
from utils import preprocess_data
# Define the tabular data for monthly water stored
monthly_water_data = {
    'month': np.arange(1, 13),  # Months from January to December
    'water_stored': [
        10.2380884146341, 5.84775914634146, 5.29252865853658,
        5.61351768292683, 4.08576097560976, 4.64988475609756,
        5.23443292682927, 6.69092073170732, 6.91169939024391,
        10.905285625, 14.673949375, 14.14763625
    ]
}
# Create the DataFrame
monthly_water_stored = pd.DataFrame(monthly_water_data)

# Load input data
input_data_path = ("/home/cedric/repos/cassini_data/uk_first"
                   "_basin/CAMELS_GB_hydromet_timeseries_31006_19701001-20150930.csv")
input_data = pd.read_csv(input_data_path)

# Ensure 'date' column is in datetime format
input_data['date'] = pd.to_datetime(input_data['date'])

# Convert all DataFrame column names to lowercase
input_data.columns = input_data.columns.str.lower()

# Select the columns you want to preprocess
input_data_to_preprocess = input_data[['precipitation', 'pet', 'temperature']]

# Preprocess the selected data
input_data_preprocessed = preprocess_data(input_data_to_preprocess)

# Merge preprocessed data back with date for filtering
input_data = pd.concat([input_data['date'], input_data_preprocessed], axis=1)

# Define the date range
start_date = '2014-11-05'
end_date = '2014-11-19'

# Filter data between the specified dates
filtered_data = input_data[(input_data['date'] >= start_date) & (input_data['date'] <= end_date)]

# Assuming the trained model and scaler are already available
best_model = load("/home/cedric/repos/cassini_data/uk_first_basin/qi_estimator.joblib")

# Assuming the input data includes necessary features for prediction
# Scaling the input data features
X_last_two_weeks = filtered_data.drop('date', axis=1)

# Perform predictions for the last two weeks
predicted_upstream_flux = best_model.predict(X_last_two_weeks)

# Create DataFrame for the predicted upstream flux for the last two weeks
start_date = pd.to_datetime('2014-11-05')
predicted_df_last_two_weeks = pd.DataFrame({
    'date': pd.date_range(start=start_date + pd.Timedelta(days=1), periods=len(predicted_upstream_flux), freq='D'),
    'upstream_flux': predicted_upstream_flux
})


# Assuming 'water_stored' data is in the 'input_data' with a 'date' column
# Subtract the downstream flux and divide by the number of days in the month

# Then continue with the script as it is:
# Get the number of days in each month
monthly_water_stored['days_in_month'] = monthly_water_stored['month'].apply(lambda x: pd.Period(f'2023-{x:02d}').days_in_month)
# Divide by the number of days in the month to get the daily average
monthly_water_stored['average_daily_water_stored'] = monthly_water_stored['water_stored'] / monthly_water_stored['days_in_month']


# Assuming 'predicted_df_last_two_weeks' and 'monthly_water_stored' DataFrames are already defined and populated.

# Mean and standard deviation of discharge_vol
mean_discharge_vol = 0.6754921488360726
std_dev_discharge_vol = 0.4567500859422541

# Process for predicted_df_last_two_weeks
predicted_df_last_two_weeks['month'] = predicted_df_last_two_weeks['date'].dt.month
dam_water_storage_month = monthly_water_stored['average_daily_water_stored'][monthly_water_stored['month']==11]
predicted_df_last_two_weeks['downstream_flux'] = predicted_df_last_two_weeks['upstream_flux'] - dam_water_storage_month.values
predicted_df_last_two_weeks['date'] = predicted_df_last_two_weeks['date'].apply(lambda d: d.replace(year=2023))

# Normalization of 'downstream_flux' or any other relevant column
predicted_df_last_two_weeks['normalized_downstream_flux'] = predicted_df_last_two_weeks['downstream_flux'].apply(
    lambda x: (x - mean_discharge_vol) / std_dev_discharge_vol)

predicted_df_last_two_weeks[['date', 'normalized_downstream_flux']].to_csv('/home/cedric/repos/cassini_data/uk_first_basin/data_for_lars/forecasting.csv')