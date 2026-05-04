import pandas as pd
import json
from pymongo import MongoClient

def setup_mongo_and_process():
    client = MongoClient("mongodb+srv://hello:hello@cluster0.cgn7dpz.mongodb.net/?appName=Cluster0") 
    db = client["traffic_project"]
    collection = db["accidents"]

    try:
        with open("nyc_collisions_final.json", "r") as file:
            data = json.load(file)
    except FileNotFoundError:
        return

    df = pd.DataFrame(data)

    df = df.dropna(subset=['on_street_name'])
    fdr_accidents = df[df['on_street_name'].str.contains("FDR", case=False)].copy()
    
    fdr_accidents['crash_date'] = fdr_accidents['crash_date'].str.split('T').str[0]
    fdr_accidents['full_crash_time'] = pd.to_datetime(fdr_accidents['crash_date'] + ' ' + fdr_accidents['crash_time'])

    fdr_accidents['hour'] = fdr_accidents['full_crash_time'].dt.hour
    fdr_accidents['day_of_week'] = fdr_accidents['full_crash_time'].dt.dayofweek

    records = fdr_accidents.to_dict(orient='records')
    if records:
        collection.insert_many(records)
    else:
        pass

if __name__ == "__main__":
    setup_mongo_and_process()