import requests
import psycopg2
import time
from datetime import datetime

# Credentials
API_KEY = "Bs4uffOtSpdO8Fj7QF2legoU1iRowti8"

# Database Configuration
DB_PARAMS = {
    "dbname": "postgres",
    "user": "postgres",
    "password": "rutgers",
    "host": "localhost",
    "port": "5432"
}

def setup_and_collect():
    conn = None
    try:
        # Connect to Docker Postgres
        conn = psycopg2.connect(**DB_PARAMS)
        cur = conn.cursor()
        
        # Ensure table exists
        cur.execute("""
            CREATE TABLE IF NOT EXISTS tomtom_traffic (
                id SERIAL PRIMARY KEY,
                current_speed INT,
                free_flow_speed INT,
                confidence FLOAT,
                captured_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)
        conn.commit()
        print("✅ PostgreSQL Table Ready.")

        # Verified working URL for FDR Drive
        url = f"https://api.tomtom.com/traffic/services/4/flowSegmentData/absolute/10/json?key={API_KEY}&point=40.7478,-73.9718"
        
        print(f"🚀 Heartbeat Collection Active. Building history every 5 minutes...")
        
        while True:
            try:
                response = requests.get(url, timeout=10)
                
                if response.status_code == 200:
                    data = response.json()
                    flow = data.get('flowSegmentData')
                    
                    if flow:
                        cur.execute(
                            "INSERT INTO tomtom_traffic (current_speed, free_flow_speed, confidence) VALUES (%s, %s, %s)",
                            (flow['currentSpeed'], flow['freeFlowSpeed'], flow['confidence'])
                        )
                        conn.commit()
                        print(f"{datetime.now().strftime('%H:%M:%S')}] Saved: {flow['currentSpeed']} mph")
                else:
                    print(f"API Error {response.status_code}: {response.text}")

            except Exception as e:
                print(f"Connection issue: {e}")

            # Interval: 5 minutes
            time.sleep(300)

    except Exception as e:
        print(f"Error: {e}")
    finally:
        if conn:
            cur.close()
            conn.close()

if __name__ == "__main__":
    setup_and_collect()