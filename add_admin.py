import sqlite3

conn = sqlite3.connect("airlink.db")
cursor = conn.cursor()

cursor.execute(
"INSERT INTO users (username,password) VALUES (?,?)",
("admin","air123")
)

conn.commit()
conn.close()

print("Admin created successfully")