import sqlite3

conn = sqlite3.connect("airlink.db")

cursor = conn.cursor()

# USERS TABLE
cursor.execute("""
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT UNIQUE,
    password TEXT
)
""")

# BOOKINGS TABLE
cursor.execute("""
CREATE TABLE IF NOT EXISTS bookings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    booking_id TEXT,
    passenger TEXT,
    source TEXT,
    destination TEXT,
    airline TEXT,
    price REAL,
    seat TEXT,
    pnr TEXT,
    date TEXT
)
""")

conn.commit()
conn.close()

print("Database Created Successfully")