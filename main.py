from flask import Flask, request, jsonify, render_template
import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, RobustScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, GRU, RepeatVector, TimeDistributed
from keras.callbacks import EarlyStopping
import plotly.io as pio
import plotly.express as px
import plotly.graph_objects as go

app = Flask(__name__)

# Load the trained model
model = tf.keras.models.load_model("fetchChallenge.h5")

@app.route('/')
def index():
    return render_template('index.html')  # Basic input form to upload data

def create_dataset(dataset, look_back=90, forecast_horizon=30):
    X, Y = [], []
    for i in range(len(dataset) - look_back - forecast_horizon + 1):
        X.append(dataset[i:(i + look_back), 0])
        Y.append(dataset[i + look_back:i + look_back + forecast_horizon, 0])
    return np.array(X), np.array(Y)

@app.route('/predict', methods=['GET'])
def predict():
    
    # Load your data and rename # Date to Date
    data = pd.read_csv('data_daily.csv')
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
    forecast_df = pd.DataFrame({'Date': forecast_dates, 'Predictions': predictions_2022})

    # If you want monthly aggregates:
    monthly_predictions = forecast_df.resample('M', on='Date').mean().reset_index()
    monthly_predictions['Date'] = monthly_predictions['Date'].apply(lambda x: x.replace(day=15))
    
    
    # return monthly_predictions.to_json(orient='records', date_format="iso")

    # Create a figure object
    fig = go.Figure()

    # Add the data for 2021 as a blue line
    fig.add_trace(go.Scatter(
        x=data['Date'],
        y=data['Receipt_Count'],
        mode='lines',
        name='Daily Receipts for 2021'
    ))

    # Overlay the predicted monthly data for 2022 as a red line
    fig.add_trace(go.Scatter(
        x=forecast_df['Date'],
        y=forecast_df['Predictions'],
        mode='lines',
        line=dict(color='red', width=2),
        name='Predicted Monthly Receipts for 2022'
    ))

    # Enhance the layout
    fig.update_layout(
        title='Daily Receipts for 2021 and Predicted Monthly Receipts for 2022',
        xaxis_title="Date",
        yaxis_title="Receipt Count",
        hovermode="x unified",
        template="plotly_dark",
    )

    # Convert the figure to JSON
    fig_json = pio.to_json(fig)

    # Return both the JSON data and the figure JSON
    return jsonify({
        'data': monthly_predictions.to_dict(orient='records'),
        'graph': fig_json
    })

if __name__ == '__main__':
    app.run(debug=True)
