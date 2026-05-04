import pandas as pd
import sqlite3
import requests
from pymongo import MongoClient

conn = sqlite3.connect("traffic_data.db")
query = "SELECT captured_at, current_speed, free_flow_speed FROM tomtom_traffic"

traffic_df = pd.read_sql(query, conn)
traffic_df['captured_at'] = pd.to_datetime(traffic_df['captured_at'])

traffic_df['is_congested'] = (traffic_df['current_speed'] < (traffic_df['free_flow_speed'] * 0.7)).astype(int)

mongo_client = MongoClient("mongodb+srv://hello:hello@cluster0.cgn7dpz.mongodb.net/?appName=Cluster0")
db = mongo_client["traffic_project"]
accidents = pd.DataFrame(list(db["accidents"].find()))

if not accidents.empty:
    crash_counts = accidents.groupby(['day_of_week', 'hour']).size().reset_index(name='historical_crash_count')
    
    traffic_df['hour'] = traffic_df['captured_at'].dt.hour
    traffic_df['day_of_week'] = traffic_df['captured_at'].dt.dayofweek
    
    merged_df = pd.merge(traffic_df, crash_counts, on=['day_of_week', 'hour'], how='left')
    merged_df['historical_crash_count'] = merged_df['historical_crash_count'].fillna(0)
else:
    merged_df = traffic_df
    merged_df['historical_crash_count'] = 0

def get_weather():
    url = "https://api.open-meteo.com/v1/forecast?latitude=40.7478&longitude=-73.9718&current_weather=true"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        return data['current_weather']['temperature'], data['current_weather']['weathercode']
    else:
        return 0, 0

temp, w_code = get_weather()
merged_df['temperature'] = temp
merged_df['weather_code'] = w_code

merged_df.to_csv("final_ai_data.csv", index=False)