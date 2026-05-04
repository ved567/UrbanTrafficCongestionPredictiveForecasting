import requests
import sqlite3 
import time
import os
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("TOMTOM_API_KEY")

def setup_and_collect():
    if not API_KEY:
        print("Error: TOMTOM_API_KEY not found in .env file.")
        return

    conn = sqlite3.connect("data/traffic_data.db")
    cur = conn.cursor()
    
    cur.execute("""
        CREATE TABLE IF NOT EXISTS tomtom_traffic (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            current_speed INTEGER,
            free_flow_speed INTEGER,
            confidence REAL,
            captured_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """)
    conn.commit()  

    url = f"https://api.tomtom.com/traffic/services/4/flowSegmentData/absolute/10/json?key={API_KEY}&point=40.7478,-73.9718"
    
    while True:
        try:
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                data = response.json()
                flow = data.get('flowSegmentData')
                
                if flow:
                    cur.execute(
                        "INSERT INTO tomtom_traffic (current_speed, free_flow_speed, confidence) VALUES (?, ?, ?)",
                        (flow['currentSpeed'], flow['freeFlowSpeed'], flow['confidence'])
                    )
                    conn.commit()
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] Saved: {flow['currentSpeed']} mph")
        except Exception as e:
            print(f"Error: {e}")

        time.sleep(300)

if __name__ == "__main__":
    setup_and_collect()