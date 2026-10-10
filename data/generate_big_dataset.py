import pandas as pd
import random
import numpy as np
from datetime import datetime

# Load existing dataset
df = pd.read_csv("enhanced_flight_data.csv")

# Indian Airlines
airlines = [
    "IndiGo",
    "Air India",
    "Vistara",
    "SpiceJet",
    "Akasa Air",
    "Air India Express",
    "Alliance Air"
]

# Indian Cities
cities = [
    "Delhi","Mumbai","Bangalore","Hyderabad","Chennai","Kolkata",
    "Ahmedabad","Pune","Goa","Jaipur","Lucknow","Chandigarh",
    "Visakhapatnam","Coimbatore","Nagpur","Indore","Bhopal",
    "Patna","Varanasi","Guwahati","Ranchi","Srinagar",
    "Amritsar","Mangalore","Tirupati","Madurai","Surat",
    "Vadodara","Raipur","Dehradun"
]

synthetic_rows = []

for _ in range(150000):

    airline = random.choice(airlines)

    source = random.choice(cities)
    destination = random.choice(cities)

    if source == destination:
        continue

    month = random.randint(1,12)
    day = random.randint(1,28)

    stops = random.randint(0,2)

    duration_hours = random.randint(1,5)
    duration_minutes = random.randint(0,59)

    days_before = random.randint(1,60)

    festival = 1 if month in [10,11] else 0
    ipl = 1 if month in [4,5] else 0

    date_obj = datetime(2019,month,day)
    weekday = date_obj.weekday()

    # Base price
    price = 2500

    # Airline pricing
    if airline == "Vistara":
        price += 2200
    elif airline == "Air India":
        price += 1800
    elif airline == "IndiGo":
        price += 1200
    elif airline == "SpiceJet":
        price += 900
    elif airline == "Akasa Air":
        price += 850
    elif airline == "Air India Express":
        price += 700
    else:
        price += 600

    # Stops impact
    price += stops * 700

    # Duration impact
    price += duration_hours * 400
    price += duration_minutes * 5

    # Booking window impact
    if days_before <= 3:
        price += 2000
    elif days_before <= 7:
        price += 1200
    elif days_before <= 15:
        price += 600

    # Festival surge
    if festival:
        price += 3000

    # IPL surge
    if ipl:
        price += 2000

    # Weekend surge
    if weekday in [4,5,6]:
        price += 1200

    # Random fluctuation
    price += random.randint(-600,600)

    synthetic_rows.append([
        airline,
        source,
        destination,
        stops,
        price,
        day,
        month,
        weekday,
        duration_hours,
        duration_minutes,
        festival,
        ipl,
        days_before
    ])

columns = [
    "Airline",
    "Source",
    "Destination",
    "Total_Stops",
    "Price",
    "Journey_Day",
    "Journey_Month",
    "Journey_Weekday",
    "Duration_Hours",
    "Duration_Minutes",
    "Festival",
    "IPL",
    "Days_Before"
]

synthetic_df = pd.DataFrame(synthetic_rows, columns=columns)

# Combine with original dataset
final_df = pd.concat([df, synthetic_df], ignore_index=True)

# Save new dataset
final_df.to_csv("big_flight_dataset.csv", index=False)

print("Dataset created successfully")
print("Total rows:", final_df.shape)