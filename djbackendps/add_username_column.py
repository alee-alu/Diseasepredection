import os
import sqlite3

# Get the path to the SQLite database
db_path = os.path.join(os.path.dirname(__file__), 'db.sqlite3')

# Connect to the database
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# Check if the column already exists
cursor.execute("PRAGMA table_info(predictions_predictionrecord)")
columns = cursor.fetchall()
column_names = [column[1] for column in columns]

if 'username' not in column_names:
    # Add the username column
    cursor.execute("ALTER TABLE predictions_predictionrecord ADD COLUMN username VARCHAR(50)")
    conn.commit()
    print("Username column added successfully")
else:
    print("Username column already exists")

# Close the connection
conn.close()
