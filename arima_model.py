import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error
import joblib

# Load cleaned dataset
df = pd.read_csv("data/cleaned_flight_data.csv")

# Create proper date column again for time series
df['Date'] = pd.to_datetime({
    'year': 2019,
    'month': df['Journey_Month'],
    'day': df['Journey_Day']
})

# Sort by date
df = df.sort_values("Date")

# Take daily average price
daily_prices = df.groupby("Date")["Price"].mean()

# Split train/test (80/20)
train_size = int(len(daily_prices) * 0.8)
train = daily_prices[:train_size]
test = daily_prices[train_size:]

# Train ARIMA
model = ARIMA(train, order=(5,1,0))
model_fit = model.fit()

# Predict
predictions = model_fit.forecast(steps=len(test))

# Evaluate
mae = mean_absolute_error(test, predictions)

print("ARIMA MAE:", mae)

# Save model
joblib.dump(model_fit, "models/arima_model.pkl")

print("ARIMA model saved successfully!")

# Plot
plt.figure(figsize=(10,5))
plt.plot(test.index, test, label="Actual")
plt.plot(test.index, predictions, label="Predicted")
plt.legend()
plt.title("ARIMA Prediction vs Actual")
plt.show()