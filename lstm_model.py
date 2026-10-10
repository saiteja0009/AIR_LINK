import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import joblib

# Load cleaned dataset
df = pd.read_csv("data/cleaned_flight_data.csv")

# Create Date again
df['Date'] = pd.to_datetime({
    'year': 2019,
    'month': df['Journey_Month'],
    'day': df['Journey_Day']
})

df = df.sort_values("Date")

daily_prices = df.groupby("Date")["Price"].mean()

# Scale data
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(daily_prices.values.reshape(-1,1))

# Create sequences
X = []
y = []

sequence_length = 10

for i in range(sequence_length, len(scaled_data)):
    X.append(scaled_data[i-sequence_length:i])
    y.append(scaled_data[i])

X, y = np.array(X), np.array(y)

# Train/Test split
train_size = int(len(X)*0.8)

X_train = X[:train_size]
X_test = X[train_size:]

y_train = y[:train_size]
y_test = y[train_size:]

# Build LSTM Model
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(X.shape[1],1)))
model.add(LSTM(50))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mse')

model.fit(X_train, y_train, epochs=20, batch_size=16)

# Predict
predictions = model.predict(X_test)
predictions = scaler.inverse_transform(predictions)
y_test_actual = scaler.inverse_transform(y_test)

mae = mean_absolute_error(y_test_actual, predictions)

print("LSTM MAE:", mae)

# Save model
model.save("models/lstm_model.h5")
joblib.dump(scaler, "models/lstm_scaler.pkl")

print("LSTM model saved successfully!")