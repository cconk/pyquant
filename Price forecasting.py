import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller

# Fetch SPY historical data
spy = yf.download('SPY', start='2010-01-01', end='2024-10-14')
spy['Adj Close'].plot(title='SPY Adjusted Closing Price')
#plt.show()

# Prepare data
prices = spy['Adj Close'].dropna()

def adf_test(series):
    result = adfuller(series)
    print(f'ADF Statistic: {result[0]}')
    print(f'p-value: {result[1]}')
    
# ADF test on the SPY data
adf_test(prices)

prices_diff = prices.diff().dropna()
adf_test(prices_diff)

# Fit ARIMA model
model = ARIMA(prices_diff, order=(5, 1, 0))  # You can try different (p,d,q) values
arima_model = model.fit()

# Print summary of the model
print(arima_model.summary())

# Forecast future prices
forecastDifferences = arima_model.forecast(steps=10)  # Forecast for next 10 days
print(forecastDifferences)

# Convert differences to prices
last_price = prices[-1]
forecast = [last_price + sum(forecastDifferences[:i]) for i in range(1, len(forecastDifferences) + 1)]
print(forecast)

# Plot the forecasted results
plt.plot(prices_diff.index, prices_diff, label='Historical Prices')
plt.plot(pd.date_range(start=prices_diff.index[-1], periods=11, freq='B')[1:], forecast, label='Forecasted Prices')
plt.legend()
plt.show()


