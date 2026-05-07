import requests
import sqlite3 
import time
import os
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("TOMTOM_API_KEY")

def setup_db():
    """
    Initializes the SQLite database with WAL mode and creates 
    the relational schema for sensors and traffic data.
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
            captured_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (sensor_id) REFERENCES sensors (sensor_id)
        );
    """)

    cur.execute("CREATE INDEX IF NOT EXISTS idx_captured_at ON tomtom_traffic (captured_at);")
    conn.commit()
    return conn

def get_or_create_sensor(conn, name, lat, lon):
    """
    Retrieves the sensor_id for a given sensor name, 
    or creates a new record if it does not exist.
    """
    cur = conn.cursor()
    cur.execute("SELECT sensor_id FROM sensors WHERE name = ?", (name,))
    result = cur.fetchone()
    
    if result:
        return result[0]
    else:
        cur.execute(
            "INSERT INTO sensors (name, latitude, longitude) VALUES (?, ?, ?)",
            (name, lat, lon)
        )
        conn.commit()
        return cur.lastrowid

def setup_and_collect():
    """
    Continuously collects real-time traffic flow data from the TomTom API 
    and persists it to the database at regular intervals.
    """
    if not API_KEY:
        print("Error: TOMTOM_API_KEY not found in .env file.")
        return

    conn = setup_db()
    SENSOR_NAME = "FDR Drive Point"
    LAT, LON = 40.7478, -73.9718
    sensor_id = get_or_create_sensor(conn, SENSOR_NAME, LAT, LON)
    
    url = f"https://api.tomtom.com/traffic/services/4/flowSegmentData/absolute/10/json?key={API_KEY}&point={LAT},{LON}"
    
    print(f"Starting collection for sensor '{SENSOR_NAME}' (ID: {sensor_id})...")
    
    cur = conn.cursor()
    while True:
        try:
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                data = response.json()
                flow = data.get('flowSegmentData')
                
                if flow:
                    cur.execute("""
                        INSERT INTO tomtom_traffic (sensor_id, current_speed, free_flow_speed, confidence)
                        VALUES (?, ?, ?, ?)
                    """, (sensor_id, flow['currentSpeed'], flow['freeFlowSpeed'], flow['confidence']))
                    conn.commit()
                    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Saved: {flow['currentSpeed']} mph")
            else:
                print(f"API Error: Status {response.status_code} - {response.text}")
        except Exception as e:
            print(f"Collection Error: {e}")

        time.sleep(300)

if __name__ == "__main__":
    setup_and_collect()
