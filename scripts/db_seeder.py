import sqlite3
import random
import os
from datetime import datetime, timedelta

def setup_db():
    """
    Initializes the database schema and WAL mode to support 
    synthetic data generation.
    """
    os.makedirs("data", exist_ok=True)
    conn = sqlite3.connect("data/traffic_data.db")
    conn.execute("PRAGMA journal_mode=WAL;")
    
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS sensors (
            sensor_id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL UNIQUE,
            latitude REAL NOT NULL,
            longitude REAL NOT NULL
        );
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS tomtom_traffic (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            sensor_id INTEGER,
            current_speed INTEGER,
            free_flow_speed INTEGER,
            confidence REAL,
            captured_at TIMESTAMP,
            FOREIGN KEY (sensor_id) REFERENCES sensors (sensor_id)
        );
    """)
    cur.execute("CREATE INDEX IF NOT EXISTS idx_captured_at ON tomtom_traffic (captured_at);")
    conn.commit()
    return conn

def get_or_create_sensor(conn):
    """
    Ensures the default FDR Drive sensor exists in the sensors table.
    """
    cur = conn.cursor()
    name, lat, lon = "FDR Drive Point", 40.7478, -73.9718
    cur.execute("SELECT sensor_id FROM sensors WHERE name = ?", (name,))
    result = cur.fetchone()
    if result:
        return result[0]
    cur.execute("INSERT INTO sensors (name, latitude, longitude) VALUES (?, ?, ?)", (name, lat, lon))
    conn.commit()
    return cur.lastrowid

def calculate_realistic_speed(dt):
    """
    Calculates a synthetic traffic speed based on time-of-day 
    and day-of-week patterns to simulate realistic NYC congestion.
    """
    hour = dt.hour
    is_weekday = dt.weekday() < 5
    
    if is_weekday:
        if 7 <= hour <= 10:  
            speed = random.randint(15, 30)
        elif 15 <= hour <= 19:  
            speed = random.randint(10, 25)
        elif 22 <= hour or hour <= 5: 
            speed = random.randint(45, 55)
        else:
            speed = random.randint(35, 45)
    else:
        if 11 <= hour <= 20:
            speed = random.randint(35, 50)
        else:
            speed = random.randint(48, 55)
            
    return speed

def seed_data(num_rows=5000):
    """
    Generates and inserts a specified number of synthetic 
    traffic records into the database in efficient batches.
    """
    conn = setup_db()
    sensor_id = get_or_create_sensor(conn)
    cur = conn.cursor()
    
    print(f"Generating {num_rows} rows of synthetic traffic data...")
    
    current_time = datetime.now()
    batch_size = 500
    data_to_insert = []
    
    for i in range(num_rows):
        captured_at = current_time - timedelta(minutes=15 * i)
        current_speed = calculate_realistic_speed(captured_at)
        free_flow_speed = 55
        confidence = round(random.uniform(0.8, 1.0), 2)
        
        data_to_insert.append((
            sensor_id, 
            current_speed, 
            free_flow_speed, 
            confidence, 
            captured_at.strftime('%Y-%m-%d %H:%M:%S')
        ))
        
        if len(data_to_insert) >= batch_size:
            cur.executemany("""
                INSERT INTO tomtom_traffic (sensor_id, current_speed, free_flow_speed, confidence, captured_at)
                VALUES (?, ?, ?, ?, ?)
            """, data_to_insert)
            conn.commit()
            data_to_insert = []
            print(f"Inserted {i + 1} rows...")

    if data_to_insert:
        cur.executemany("""
            INSERT INTO tomtom_traffic (sensor_id, current_speed, free_flow_speed, confidence, captured_at)
            VALUES (?, ?, ?, ?, ?)
        """, data_to_insert)
        conn.commit()

    print("Seeding complete.")
    conn.close()

if __name__ == "__main__":
    seed_data()
