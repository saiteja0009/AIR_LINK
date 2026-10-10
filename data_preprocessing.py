import pandas as pd
import numpy as np

# Load Excel dataset
df = pd.read_excel("data/flight_data.xlsx")

print("Original Shape:", df.shape)
print(df.head())

# -----------------------------
# 1️⃣ Handle Date Column
# -----------------------------

df['Date_of_Journey'] = pd.to_datetime(df['Date_of_Journey'], dayfirst=True)

df['Journey_Day'] = df['Date_of_Journey'].dt.day
df['Journey_Month'] = df['Date_of_Journey'].dt.month
df['Journey_Weekday'] = df['Date_of_Journey'].dt.dayofweek

df.drop('Date_of_Journey', axis=1, inplace=True)

# -----------------------------
# 2️⃣ Handle Duration
# -----------------------------

# -----------------------------
# 2️⃣ Handle Duration (FIXED)
# -----------------------------

df['Duration'] = df['Duration'].str.replace('h', ' ')
df['Duration'] = df['Duration'].str.replace('m', ' ')
df['Duration'] = df['Duration'].fillna("0 0")

duration = df['Duration'].str.split()

df['Duration_Hours'] = duration.str[0].astype(int)

# Some rows have only hours, so fill missing minutes with 0
df['Duration_Minutes'] = duration.str[1]
df['Duration_Minutes'] = df['Duration_Minutes'].fillna(0)
df['Duration_Minutes'] = df['Duration_Minutes'].astype(int)

if 'Duration' in df.columns:
    df.drop('Duration', axis=1, inplace=True)

# -----------------------------
# 3️⃣ Total Stops
# -----------------------------

df['Total_Stops'] = df['Total_Stops'].replace({
    'non-stop': 0,
    '1 stop': 1,
    '2 stops': 2,
    '3 stops': 3,
    '4 stops': 4
})

# -----------------------------
# 4️⃣ Simulated Event Features
# -----------------------------

df['Festival'] = np.where(df['Journey_Month'].isin([10, 11]), 1, 0)
df['IPL'] = np.where(df['Journey_Month'].isin([4, 5]), 1, 0)

# -----------------------------
# 5️⃣ Booking Window Simulation
# -----------------------------

df['Days_Before'] = np.random.randint(1, 30, size=len(df))

# -----------------------------
# 6️⃣ Drop Unnecessary Columns
# -----------------------------

df.drop(['Route', 'Dep_Time', 'Arrival_Time', 'Additional_Info'], axis=1, inplace=True)

# -----------------------------
# Save Cleaned Dataset
# -----------------------------

df.to_csv("data/cleaned_flight_data.csv", index=False)

print("Cleaned dataset saved successfully!")
print("New Shape:", df.shape)