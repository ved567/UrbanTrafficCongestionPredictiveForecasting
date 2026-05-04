import pandas as pd
import json
from tinydb import TinyDB

def setup_local_nosql():
    db = TinyDB("data/accidents_nosql.json")
    db.truncate()

    with open("data/nyc_collisions_final.json", "r") as file:
        data = json.load(file)

    df = pd.DataFrame(data)
    df = df.dropna(subset=['on_street_name'])
    fdr_accidents = df[df['on_street_name'].str.contains("FDR|FRANKLIN D|ROOSEVELT DR|EAST RIVER DR", case=False, na=False)].copy()

    fdr_accidents['crash_date'] = fdr_accidents['crash_date'].str.split('T').str[0]
    fdr_accidents['full_crash_time'] = pd.to_datetime(fdr_accidents['crash_date'] + ' ' + fdr_accidents['crash_time'])
    fdr_accidents['hour'] = fdr_accidents['full_crash_time'].dt.hour
    fdr_accidents['day_of_week'] = fdr_accidents['full_crash_time'].dt.dayofweek

    # Convert Timestamp objects to strings so TinyDB can save them
    fdr_accidents['full_crash_time'] = fdr_accidents['full_crash_time'].astype(str)

    records = fdr_accidents.to_dict(orient='records')
    if records:
        db.insert_multiple(records)
        print(f"Stored {len(records)} records")

if __name__ == "__main__":
    setup_local_nosql()