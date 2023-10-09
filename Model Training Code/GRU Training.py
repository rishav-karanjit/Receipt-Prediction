import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
from keras.models import Sequential
from keras.layers import Dense, GRU, RepeatVector, TimeDistributed
from keras.callbacks import EarlyStopping
import plotly.express as px
import plotly.graph_objects as go


data = pd.read_csv('data_daily.csv')
data = data.rename(columns={'# Date': 'Date'})

print(data.isnull().values.any())

# Scaling data
scaler = RobustScaler()
scaled_data = scaler.fit_transform(data['Receipt_Count'].values.reshape(-1, 1))

# Convert data to appropriate shape for GRU
def create_dataset(dataset, look_back=90, forecast_horizon=30):
    X, Y = [], []
    for i in range(len(dataset) - look_back - forecast_horizon + 1):
        X.append(dataset[i:(i + look_back), 0])
        Y.append(dataset[i + look_back:i + look_back + forecast_horizon, 0])
    return np.array(X), np.array(Y)

X, y = create_dataset(scaled_data)

# Reshape input to be [samples, time steps, features]
X = X.reshape(X.shape[0], X.shape[1], 1)
y = y.reshape(y.shape[0], y.shape[1], 1)

# GRU model
model = Sequential()
model.add(GRU(20, activation='relu', input_shape=(X.shape[1], X.shape[2])))
model.add(RepeatVector(30))  # Repeat the feature vector 30 times
model.add(GRU(10, activation='relu', return_sequences=True))
model.add(TimeDistributed(Dense(1)))
model.compile(optimizer='adam', loss='mse')

early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

model.fit(X, y, epochs=100, batch_size=1, verbose=1, callbacks=[early_stopping], validation_split=0.2)



look_back_period = 90  # Set look-back period

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

print(forecast_df)

# If you want monthly aggregates:
monthly_predictions = forecast_df.resample('M', on='Date').sum()
print(monthly_predictions)

# Create the base line plot for 2021 data
fig = px.line(data, x='Date', y='Receipt_Count', title='Daily Receipts for 2021 and Predicted Monthly Receipts for 2022',
              labels={'Receipt_Count': 'Receipt Count'},
              template="plotly_dark")

# Overlay the predicted monthly data for 2022 as a red line
fig.add_trace(go.Scatter(x=forecast_df['Date'], y=forecast_df['Predictions'], mode='lines',
                         line=dict(color='red', width=2),

                         name='Predicted Monthly Receipts for 2022'))

# Enhance the layout
fig.update_layout(showlegend=True,
                  xaxis_title="Date",
                  yaxis_title="Receipt Count",
                  hovermode="x unified")

fig.show()


# In[ ]:


model.save("fetchChallenge.h5")


# 
