from pathlib import Path
import json

import pandas as pd
import requests
from tinydb import TinyDB


DATA_DIR = Path("data")
RAW_COLLISION_FILE = DATA_DIR / "nyc_collisions_final.json"
ACCIDENT_DB_FILE = DATA_DIR / "accidents_nosql.json"

NYC_COLLISION_API_URL = "https://data.cityofnewyork.us/resource/h9gi-nx95.json"


def download_nyc_collision_data():
    """
    Downloads NYC collision records and saves them as data/nyc_collisions_final.json.
    This allows the project to run from a clean clone without manually adding the file.
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    print("data/nyc_collisions_final.json not found.")
    print("Downloading NYC collision data...")

    params = {
        "$limit": 50000,
        "$order": "crash_date DESC",
        "$select": (
            "crash_date,"
            "crash_time,"
            "on_street_name,"
            "cross_street_name,"
            "latitude,"
            "longitude,"
            "number_of_persons_injured,"
            "number_of_persons_killed,"
            "collision_id,"
            "contributing_factor_vehicle_1"
        )
    }

    response = requests.get(
        NYC_COLLISION_API_URL,
        params=params,
        timeout=60
    )

    response.raise_for_status()
    data = response.json()

    with open(RAW_COLLISION_FILE, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=2)

    print(f"Saved {len(data)} records to {RAW_COLLISION_FILE}")

    return data


def load_collision_data():
    """
    Loads the raw NYC collision file.
    If it is missing, downloads it first.
    """
    if not RAW_COLLISION_FILE.exists():
        return download_nyc_collision_data()

    with open(RAW_COLLISION_FILE, "r", encoding="utf-8") as file:
        return json.load(file)


def setup_local_nosql():
    """
    Ingests NYC collision data, filters for FDR Drive crashes,
    and stores the filtered records in TinyDB.
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    db = TinyDB(ACCIDENT_DB_FILE)
    db.truncate()

    data = load_collision_data()
    df = pd.DataFrame(data)

    if df.empty:
        print("No collision records found.")
        return

    if "on_street_name" not in df.columns:
        print("Collision data is missing on_street_name.")
        return

    df = df.dropna(subset=["on_street_name"]).copy()

    fdr_accidents = df[
        df["on_street_name"].str.contains(
            "FDR|FRANKLIN D|ROOSEVELT DR|EAST RIVER DR",
            case=False,
            na=False,
            regex=True
        )
    ].copy()

    if fdr_accidents.empty:
        print("No FDR Drive collision records found.")
        return

    fdr_accidents["crash_date"] = (
        fdr_accidents["crash_date"]
        .astype(str)
        .str.split("T")
        .str[0]
    )

    fdr_accidents["full_crash_time"] = pd.to_datetime(
        fdr_accidents["crash_date"] + " " + fdr_accidents["crash_time"].astype(str),
        errors="coerce"
    )

    fdr_accidents = fdr_accidents.dropna(subset=["full_crash_time"]).copy()

    fdr_accidents["hour"] = fdr_accidents["full_crash_time"].dt.hour
    fdr_accidents["day_of_week"] = fdr_accidents["full_crash_time"].dt.dayofweek
    fdr_accidents["full_crash_time"] = fdr_accidents["full_crash_time"].astype(str)

    for col in [
        "latitude",
        "longitude",
        "number_of_persons_injured",
        "number_of_persons_killed"
    ]:
        if col in fdr_accidents.columns:
            fdr_accidents[col] = pd.to_numeric(fdr_accidents[col], errors="coerce")

    fdr_accidents = fdr_accidents.where(pd.notna(fdr_accidents), None)

    records = fdr_accidents.to_dict(orient="records")

    if records:
        db.insert_multiple(records)

    print(f"Stored {len(records)} FDR collision records in {ACCIDENT_DB_FILE}")


if __name__ == "__main__":
    setup_local_nosql()
