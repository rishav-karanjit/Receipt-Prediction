import numpy as np
import plotly.io as pio
import plotly.graph_objects as go
import pandas as pd

def create_dataset(dataset, look_back=90, forecast_horizon=30):
    X, Y = [], []
    for i in range(len(dataset) - look_back - forecast_horizon + 1):
        X.append(dataset[i:(i + look_back), 0])
        Y.append(dataset[i + look_back:i + look_back + forecast_horizon, 0])
    return np.array(X), np.array(Y)

def createGraph(forecast_2022):
    data = pd.read_csv('./DataSet/data_daily.csv')
    data = data.rename(columns={'# Date': 'Date'})
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
        x=forecast_2022['Date'],
        y=forecast_2022['Receipt_Count'],
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
        hoverlabel=dict(
            namelength=-1  # this will display the entire name without truncating
        )
    )

    # Convert the figure to JSON
    return(pio.to_json(fig))
