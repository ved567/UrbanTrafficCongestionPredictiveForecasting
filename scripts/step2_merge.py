import pandas as pd
import sqlite3
import requests
from tinydb import TinyDB

conn = sqlite3.connect("data/traffic_data.db")
query = "SELECT captured_at, current_speed, free_flow_speed FROM tomtom_traffic"
traffic_df = pd.read_sql(query, conn)
traffic_df['captured_at'] = pd.to_datetime(traffic_df['captured_at'])
traffic_df['hour'] = traffic_df['captured_at'].dt.hour
traffic_df['day_of_week'] = traffic_df['captured_at'].dt.dayofweek

db = TinyDB("data/accidents_nosql.json")
accidents = pd.DataFrame(db.all())

if not accidents.empty:
    crash_counts = accidents.groupby(['day_of_week', 'hour']).size().reset_index(name='historical_crash_count')
    merged_df = pd.merge(traffic_df, crash_counts, on=['day_of_week', 'hour'], how='left')
    merged_df['historical_crash_count'] = merged_df['historical_crash_count'].fillna(0)
else:
    traffic_df['historical_crash_count'] = 0
    merged_df = traffic_df

def get_weather():
    try:
        url = "https://api.open-meteo.com/v1/forecast?latitude=40.7478&longitude=-73.9718&current_weather=true"
        data = requests.get(url).json()
        return data['current_weather']['temperature'], data['current_weather']['weathercode']
    except:
        return 20.0, 0

temp, w_code = get_weather()
merged_df['temperature'] = temp
merged_df['weather_code'] = w_code
merged_df.to_csv("data/final_ai_data.csv", index=False)