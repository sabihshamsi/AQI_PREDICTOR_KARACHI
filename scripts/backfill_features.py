from __future__ import annotations

import argparse
from datetime import datetime, timedelta
from pathlib import Path
import sys
from zoneinfo import ZoneInfo

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.hopsworks_utils import insert_features
from src.open_meteo import KARACHI_TIMEZONE, fetch_open_meteo


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill Karachi AQI features to Hopsworks")

    today = datetime.now(ZoneInfo(KARACHI_TIMEZONE)).date()
    default_start = today - timedelta(days=365)

    parser.add_argument("--start-date", default=str(default_start), help="YYYY-MM-DD")
    parser.add_argument("--end-date", default=str(today), help="YYYY-MM-DD")

    args = parser.parse_args()

    start = datetime.strptime(args.start_date, "%Y-%m-%d").date()
    end = datetime.strptime(args.end_date, "%Y-%m-%d").date()

    if start >= end:
        raise ValueError("start-date must be before end-date")

    df = fetch_open_meteo(str(start), str(end))
    df["date"] = pd.to_datetime(df["date"])

    print("\n=== BACKFILL STATS ===")
    print(f"Rows: {len(df)}")
    print(f"Start: {df['date'].min()}")
    print(f"End: {df['date'].max()}")
    print("======================\n")

    if df.empty:
        raise ValueError("No data fetched from Open-Meteo")

    df["date"] = pd.to_datetime(df["date"]).dt.date

    insert_features(df)

    print(f"Inserted {len(df)} daily rows into Hopsworks")


if __name__ == "__main__":
    main()
