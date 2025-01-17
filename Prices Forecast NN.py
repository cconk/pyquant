import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import yfinance as yf

# Prepare data
spy = yf.download('SPY', start='2010-01-01', end='2024-10-14')
data = spy['Adj Close'].values
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data.reshape(-1, 1))

# Create sequences
def create_sequences(data, sequence_length=60):
    sequences = []
    labels = []
    for i in range(len(data) - sequence_length):
        sequences.append(data[i:i + sequence_length])
        labels.append(data[i + sequence_length])
    return np.array(sequences), np.array(labels)

# Create sequences with 60 time steps
X, y = create_sequences(scaled_data, 60)

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Build the LSTM model
model = tf.keras.Sequential([
    tf.keras.layers.LSTM(100, return_sequences=True, input_shape=(X_train.shape[1], 1)),
    tf.keras.layers.LSTM(50, return_sequences=False),
    tf.keras.layers.Dense(25),
    tf.keras.layers.Dense(1)
])

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(X_train, y_train, batch_size=64, epochs=10)

# Predict on the test data
predictions = model.predict(X_test)
predictions = scaler.inverse_transform(predictions)  # Convert back to original scale

# Plot the results
plt.figure(figsize=(10,6))
plt.plot(spy.index[-len(y_test):], scaler.inverse_transform(y_test.reshape(-1, 1)), color='blue', label='Actual SPY Price')
plt.plot(spy.index[-len(predictions):], predictions, color='red', label='Predicted SPY Price')
plt.title('SPY Price Prediction')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.show()

# Get the last 60 days of data from the test set
last_sequence = X_test[-1]  # Last sequence in the test set
predicted_next_days = []

# Predict one day at a time and append the prediction to the list
for _ in range(10):
    next_prediction = model.predict(last_sequence.reshape(1, -1, 1))
    predicted_next_days.append(next_prediction[0][0])
    
    # Update the last_sequence by appending the new prediction and removing the oldest step
    last_sequence = np.append(last_sequence[1:], next_prediction, axis=0)

# Rescale predictions back to original price scale
predicted_next_days = scaler.inverse_transform(np.array(predicted_next_days).reshape(-1, 1))

# Print the predicted prices for the next 10 days
print("Predicted SPY Prices for the next 10 days:")
print(predicted_next_days)

# Print the predicted prices in a readable format
for i, price in enumerate(predicted_next_days, 1):
    print(f"Day {i}: {price[0]}")