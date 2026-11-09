import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

df = yf.download("^NSEI", start="2010-01-01", end="2024-01-01")
df = df[['Close']]
df.dropna(inplace=True)

scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(df)

def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(seq_length, len(data)):
        X.append(data[i - seq_length:i, 0])
        y.append(data[i, 0])
    return np.array(X), np.array(y)

seq_length = 60
X, y = create_sequences(scaled_data, seq_length)
X = np.reshape(X, (X.shape[0], X.shape[1], 1))

model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(X.shape[1], 1)))
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(units=1))

model.compile(optimizer='adam', loss='mean_squared_error')
model.summary()
history = model.fit(X, y, epochs=20, batch_size=32)

last_sequence = scaled_data[-seq_length:]
last_sequence = np.reshape(last_sequence, (1, seq_length, 1))

predicted_price = model.predict(last_sequence)
predicted_price = scaler.inverse_transform(predicted_price)

print(f"Predicted Next Day Closing Price: ₹{predicted_price[0][0]:.2f}")

train_predict = model.predict(X)
train_predict = scaler.inverse_transform(train_predict)

y_actual = scaler.inverse_transform(y.reshape(-1, 1))

plt.figure(figsize=(12, 6))
plt.plot(y_actual, label='Actual')
plt.plot(train_predict, label='Predicted')
plt.title('NIFTY 50 Price Prediction using LSTM')
plt.xlabel('Days')
plt.ylabel('Price (INR)')
plt.legend()
plt.show()
