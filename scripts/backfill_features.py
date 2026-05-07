from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import sys

import pandas as pd
import requests

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.hopsworks_utils import insert_features

KARACHI_LAT = 24.8607
KARACHI_LON = 67.0011


def fetch_open_meteo(start_date: str, end_date: str) -> pd.DataFrame:
    url = "https://air-quality-api.open-meteo.com/v1/air-quality"
    params = {
        "latitude": KARACHI_LAT,
        "longitude": KARACHI_LON,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": ",".join(
            [
                "pm2_5",
                "pm10",
                "carbon_monoxide",
                "nitrogen_dioxide",
                "sulphur_dioxide",
                "ozone",
                "aerosol_optical_depth",
                "dust",
                "uv_index",
            ]
        ),
        "timezone": "Asia/Karachi",
    }

    response = requests.get(url, params=params, timeout=60)
    response.raise_for_status()
    payload = response.json()

    hourly = payload["hourly"]
    df = pd.DataFrame(hourly)
    df["time"] = pd.to_datetime(df["time"])

    daily = (
        df.set_index("time")
        .resample("D")
        .mean(numeric_only=True)
        .reset_index()
        .rename(columns={"time": "date"})
    )
    daily["date"] = pd.to_datetime(daily["date"]).dt.date
    return daily


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill Karachi AQI features to Hopsworks")
    parser.add_argument("--start-date", required=True, help="YYYY-MM-DD")
    parser.add_argument("--end-date", required=True, help="YYYY-MM-DD")
    args = parser.parse_args()

    datetime.strptime(args.start_date, "%Y-%m-%d")
    datetime.strptime(args.end_date, "%Y-%m-%d")

    df = fetch_open_meteo(args.start_date, args.end_date)
    insert_features(df)
    print(f"Inserted {len(df)} daily rows from {args.start_date} to {args.end_date}")


if __name__ == "__main__":
    main()
