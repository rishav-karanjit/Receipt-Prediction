import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import RobustScaler
from helper.utils import create_dataset
from prophet import Prophet
import pickle
import numpy as np

model = tf.keras.models.load_model("fetchChallenge.h5")

def predictProphetModel():
    # Load data and rename columns
    data = pd.read_csv('./DataSet/data_daily.csv')
    data = data.rename(columns={'# Date': 'ds', 'Receipt_Count': 'y'})
    data['ds'] = pd.to_datetime(data['ds'])
    
    with open('prophet_model.pkl', 'rb') as file:
        model = pickle.load(file)
    
    # Create future dataframe
    future = model.make_future_dataframe(periods=365)  # for a year's prediction

    # Predict
    forecast = model.predict(future)

    # Extract forecasted values for 2022
    forecast_2022 = forecast[forecast['ds'].dt.year == 2022][['ds', 'yhat']]
    forecast_2022['yhat'] = forecast_2022['yhat'].round(2)
    forecast_2022 = forecast_2022.rename(columns={"ds": "Date", "yhat": "Receipt_Count"})

    # Monthly aggregates
    monthly_predictions = forecast_2022.resample('M', on='Date').mean().reset_index().round(2)
    monthly_predictions['Date'] = monthly_predictions['Date'].apply(lambda x: x.replace(day=15))

    data = data.rename(columns={'ds': 'Date', 'y': 'Receipt_Count'})

    return monthly_predictions, forecast_2022
def predictGRUModel():
    # Load your data and rename # Date to Date
    data = pd.read_csv('DataSet/data_daily.csv')
    data = data.rename(columns={'# Date': 'Date'})

    scaler = RobustScaler()
    scaled_data = scaler.fit_transform(data['Receipt_Count'].values.reshape(-1, 1))

    X, y = create_dataset(scaled_data)

    # Reshape input to be [samples, time steps, features]
    X = X.reshape(X.shape[0], X.shape[1], 1)
    y = y.reshape(y.shape[0], y.shape[1], 1)

    look_back_period = 90  # Set your desired look-back period

    predictions_2022 = []

    # Start with the last 'look_back_period' days of 2021
    last_data = scaled_data[-look_back_period:]


    for i in range(13):  # 13 chunks of predictions
        forecast = model.predict(last_data.reshape(1, look_back_period, 1))
        forecast_original = scaler.inverse_transform(forecast[0])

        if i == 12:  # On the 13th loop, only take the first 5 days
            predictions_2022.extend(forecast_original[:5].flatten())
        else:
            predictions_2022.extend(forecast_original.flatten())

        # Append the forecasted values to our 'last_data' and use the most recent 'look_back_period' for the next prediction
        last_data = np.vstack((last_data[30:], scaler.transform(forecast_original)))

    # Ensure we have 365 days of predictions
    assert len(predictions_2022) == 365

    # Convert to DataFrame
    forecast_dates = pd.date_range(start="2022-01-01", end="2022-12-31", freq='D')
    forecast_df = pd.DataFrame({'Date': forecast_dates, 'Receipt_Count': predictions_2022})

    # If you want monthly aggregates:
    monthly_predictions = forecast_df.resample('M', on='Date').mean().reset_index()
    monthly_predictions['Date'] = monthly_predictions['Date'].apply(lambda x: x.replace(day=15))

    return monthly_predictions, forecast_df