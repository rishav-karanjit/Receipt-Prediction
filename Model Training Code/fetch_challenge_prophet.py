# Import necessary libraries
import pandas as pd
from prophet import Prophet
import pickle
# Load data and rename columns
data = pd.read_csv('./DataSet/data_daily.csv')
data = data.rename(columns={'# Date': 'Date', 'Receipt_Count': 'y'})
data['ds'] = pd.to_datetime(data['Date'])

# Drop the 'Date' column as we have 'ds' now
data = data.drop(columns=['Date'])

# Initialize Prophet model
model = Prophet(daily_seasonality=True)

# Fit model
model.fit(data)

# Create future dataframe
future = model.make_future_dataframe(periods=365)  # for a year's prediction

# Predict
forecast = model.predict(future)

# Extract forecasted values for 2022
forecast_2022 = forecast[forecast['ds'].dt.year == 2022][['ds', 'yhat']]

print(forecast_2022)

# Monthly aggregates
forecast_2022['month'] = forecast_2022['ds'].dt.to_period('M')
monthly_predictions = forecast_2022.groupby('month')['yhat'].sum()
print(monthly_predictions)

with open('prophet_model.pkl', 'wb') as file:
    pickle.dump(model, file)

import plotly.express as px
import plotly.graph_objects as go

# Create the base line plot for 2021 data
fig = px.line(data, x='ds', y='y', title='Daily Receipts for 2021 and Predicted Monthly Receipts for 2022',
              labels={'Receipt_Count': 'Receipt Count'},
              template="plotly_dark")

# Overlay the predicted monthly data for 2022 as a red line
fig.add_trace(go.Scatter(x=forecast_2022['ds'], y=forecast_2022['yhat'], mode='lines',
                         line=dict(color='red', width=2),

                         name='Predicted Monthly Receipts for 2022'))

# Enhance the layout
fig.update_layout(showlegend=True,
                  xaxis_title="Date",
                  yaxis_title="Receipt Count",
                  hovermode="x unified")

fig.show()