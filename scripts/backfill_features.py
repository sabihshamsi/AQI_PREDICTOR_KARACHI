from __future__ import annotations

import argparse
from datetime import datetime, timedelta
from pathlib import Path
import sys
from zoneinfo import ZoneInfo

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.hopsworks_utils import insert_features
from src.open_meteo import KARACHI_TIMEZONE, fetch_open_meteo


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill Karachi AQI features to Hopsworks")
    today = datetime.now(ZoneInfo(KARACHI_TIMEZONE)).date()
    default_start = today - timedelta(days=7)
    parser.add_argument("--start-date", default=str(default_start), help="YYYY-MM-DD")
    parser.add_argument("--end-date", default=str(today), help="YYYY-MM-DD")
    args = parser.parse_args()

    datetime.strptime(args.start_date, "%Y-%m-%d")
    datetime.strptime(args.end_date, "%Y-%m-%d")

    df = fetch_open_meteo(args.start_date, args.end_date)
    insert_features(df)
    print(f"Inserted {len(df)} daily rows from {args.start_date} to {args.end_date}")


if __name__ == "__main__":
    main()
