import pandas as pd
import numpy as np
import random
from datetime import datetime

# Load original cleaned dataset
df = pd.read_csv("data/cleaned_flight_data.csv")

synthetic_rows = []

# 🔥 Major Indian Cities (30+)
cities = [
    "Delhi", "Mumbai", "Kolkata", "Chennai", "Bangalore",
    "Hyderabad", "Ahmedabad", "Pune", "Jaipur", "Lucknow",
    "Chandigarh", "Goa", "Visakhapatnam", "Coimbatore",
    "Nagpur", "Indore", "Bhopal", "Patna", "Varanasi",
    "Guwahati", "Ranchi", "Srinagar", "Amritsar",
    "Mangalore", "Tirupati", "Madurai", "Surat",
    "Vadodara", "Raipur", "Dehradun"
]

# 🔥 Expanded Indian Airlines
airlines = [
    "IndiGo",
    "Air India",
    "Vistara",
    "SpiceJet",
    "Akasa Air",
    "Air India Express",
    "Alliance Air"
]

for _ in range(40000):  # Increased dataset size further

    airline = random.choice(airlines)

    # Ensure source and destination are not same
    source = random.choice(cities)
    destination = random.choice(cities)
    while destination == source:
        destination = random.choice(cities)

    month = random.randint(1, 12)
    day = random.randint(1, 28)

    stops = random.randint(0, 2)
    duration_hours = random.randint(1, 6)
    duration_minutes = random.randint(0, 59)

    festival = 1 if month in [10, 11] else 0
    ipl = 1 if month in [4, 5] else 0

    days_before = random.randint(1, 30)

    # ------------------------------------
    # 🎯 ADVANCED REALISTIC PRICE LOGIC
    # ------------------------------------

    base_price = 2500

    # 🟢 Airline Tier Pricing
    airline_tier = {
        "Vistara": 2200,
        "Air India": 1800,
        "IndiGo": 1200,
        "SpiceJet": 900,
        "Akasa Air": 850,
        "Air India Express": 700,
        "Alliance Air": 600
    }

    base_price += airline_tier.get(airline, 800)

    # 🟢 Stops effect
    base_price += stops * 750

    # 🟢 Duration effect
    base_price += duration_hours * 450
    base_price += duration_minutes * 6

    # 🟢 Booking window effect
    if days_before <= 3:
        base_price += 2200
    elif days_before <= 7:
        base_price += 1200
    elif days_before <= 15:
        base_price += 500

    # 🟢 Festival surge
    if festival:
        base_price += 3000

    # 🟢 IPL surge
    if ipl:
        base_price += 2000

    # 🟢 Weekend Surge
    date_obj = datetime(2019, month, day)
    weekday = date_obj.weekday()
    if weekday in [4, 5, 6]:  # Fri, Sat, Sun
        base_price += 1200

    # 🟢 Peak travel months
    if month in [5, 6, 12]:
        base_price += 1800

    # 🟢 Random market fluctuation
    base_price += random.randint(-700, 700)

    synthetic_rows.append([
        airline, source, destination,
        stops, duration_hours, duration_minutes,
        month, day, festival, ipl, days_before,
        base_price
    ])

columns = [
    "Airline", "Source", "Destination",
    "Total_Stops", "Duration_Hours", "Duration_Minutes",
    "Journey_Month", "Journey_Day",
    "Festival", "IPL", "Days_Before",
    "Price"
]

synthetic_df = pd.DataFrame(synthetic_rows, columns=columns)

# Combine with real dataset
final_df = pd.concat([df, synthetic_df], ignore_index=True)

final_df.to_csv("data/enhanced_flight_data.csv", index=False)

print("✅ Synthetic data generated successfully!")
print("📊 New Dataset Size:", final_df.shape)