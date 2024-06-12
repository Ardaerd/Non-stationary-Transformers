import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import numpy as np
def arima_result(dataa):
    # Load your dataset
    data = dataa

    # Convert 'date' column to datetime
    data['date'] = pd.to_datetime(data['date'], format='%Y/%m/%d %H:%M')

    # Set the 'date' column as the index
    data.set_index('date', inplace=True)
    data.index.freq = 'D'

    # Determine the size of your training set
    train_size = int(len(data) * 0.8)  # You can adjust the split ratio as needed

    # Split the data into training and testing sets
    train_data = data.iloc[:train_size]
    test_data = data.iloc[train_size:]

    ot_train_series = train_data['OT']
    # Fit ARIMA model
    arima_order = (2, 1, 2)
    model = ARIMA(ot_train_series, order=arima_order)
    model_fit = model.fit()

    # Forecast using ARIMA
    steps = len(test_data) # Forecasting steps, adjust as necessary
    arima_forecast = model_fit.forecast(steps=steps)
    arima_forecast = np.array(arima_forecast)

    return arima_forecast