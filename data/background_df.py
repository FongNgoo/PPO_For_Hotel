"""
Google Trends (DAILY) + Portugal Holidays
Use case: Hotel / Tourism Demand Forecasting (Algarve, Portugal)
Author: Optimized for Thesis / PPO Pipeline
"""

# ======================================================
# 0. IMPORTS
# ======================================================
from pytrends.request import TrendReq
from pytrends.exceptions import TooManyRequestsError

import pandas as pd
import holidays
import time
import random
import os

from datetime import timedelta


# ======================================================
# 1. CONFIG
# ======================================================
START_DATE = "2013-06-23"
END_DATE = "2017-09-01"

KEYWORDS = [
    "resort algarve",
    "algarve hotel",
    "algarve tourism",
    "algarve vacation"
]

GEO = "PT"

CHUNK_DAYS = 60          # safer than 90
MAX_RETRIES = 5
SLEEP_RANGE = (5, 10)    # seconds
CACHE_DIR = "cache"
CACHE_FILE = f"{CACHE_DIR}/google_trends_algarve_daily.csv"


# ======================================================
# 2. GOOGLE TRENDS (DAILY + RATE LIMIT SAFE)
# ======================================================
def get_google_trends_daily(
    keywords,
    geo,
    start_date,
    end_date,
    chunk_days=60,
    max_retries=5
):
    pytrends = TrendReq(hl="en-US", tz=0)

    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)

    all_dfs = []

    while start < end:
        chunk_end = min(start + timedelta(days=chunk_days), end)
        timeframe = f"{start.date()} {chunk_end.date()}"

        print(f"📥 Fetching Google Trends: {timeframe}")

        success = False
        attempt = 0

        while not success and attempt < max_retries:
            try:
                pytrends.build_payload(
                    kw_list=keywords,
                    geo=geo,
                    timeframe=timeframe
                )

                df = pytrends.interest_over_time()

                if not df.empty:
                    df = df.reset_index()
                    if "isPartial" in df.columns:
                        df = df.drop(columns=["isPartial"])
                    all_dfs.append(df)

                success = True

                sleep_time = random.uniform(*SLEEP_RANGE)
                print(f"⏱ Sleeping {sleep_time:.1f}s")
                time.sleep(sleep_time)

            except TooManyRequestsError:
                attempt += 1
                wait = 30 * attempt
                print(f"⚠️ 429 detected – retry {attempt}/{max_retries} after {wait}s")
                time.sleep(wait)

        if not success:
            print(f"❌ Skipped chunk: {timeframe}")

        start = chunk_end + timedelta(days=1)

    if not all_dfs:
        raise ValueError("Google Trends returned empty dataframe")

    df_final = (
        pd.concat(all_dfs)
        .drop_duplicates(subset=["date"])
        .sort_values("date")
        .reset_index(drop=True)
    )

    return df_final


# ======================================================
# 3. PORTUGAL HOLIDAYS + TYPE
# ======================================================
def add_portugal_holidays(df):
    years = range(
        df["date"].dt.year.min(),
        df["date"].dt.year.max() + 1
    )

    pt_holidays = holidays.Portugal(years=years)

    df["is_holiday_pt"] = df["date"].isin(pt_holidays)
    df["holiday_name"] = df["date"].map(pt_holidays)

    def classify_holiday(name):
        if pd.isna(name):
            return "none"
        name = name.lower()
        if "natal" in name or "christmas" in name:
            return "religious"
        if "páscoa" in name or "easter" in name:
            return "religious"
        if "republic" in name or "liberty" in name:
            return "national"
        return "public"

    df["holiday_type"] = df["holiday_name"].apply(classify_holiday)

    return df


# ======================================================
# 4. CALENDAR FEATURES
# ======================================================
def add_calendar_features(df, keywords):
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["day"] = df["date"].dt.day
    df["weekday"] = df["date"].dt.weekday
    df["is_weekend"] = df["weekday"].isin([5, 6])

    # Trend aggregation (recommended for PPO)
    df["trend_mean"] = df[keywords].mean(axis=1)
    df["trend_max"] = df[keywords].max(axis=1)

    holiday_dates = set(df.loc[df["is_holiday_pt"], "date"])

    df["is_before_holiday"] = df["date"].apply(
        lambda x: (x + timedelta(days=1)) in holiday_dates
    )
    df["is_after_holiday"] = df["date"].apply(
        lambda x: (x - timedelta(days=1)) in holiday_dates
    )

    return df


# ======================================================
# 5. MAIN PIPELINE (WITH CACHE)
# ======================================================
def main():
    os.makedirs(CACHE_DIR, exist_ok=True)

    if os.path.exists(CACHE_FILE):
        print("✅ Loading cached Google Trends data...")
        trend_df = pd.read_csv(CACHE_FILE, parse_dates=["date"])
    else:
        print("⬇️ Downloading DAILY Google Trends data...")
        trend_df = get_google_trends_daily(
            KEYWORDS,
            GEO,
            START_DATE,
            END_DATE,
            CHUNK_DAYS,
            MAX_RETRIES
        )

        trend_df.to_csv(CACHE_FILE, index=False)
        print(f"💾 Cached: {CACHE_FILE}")

    print("📅 Adding Portugal holidays...")
    trend_df = add_portugal_holidays(trend_df)

    print("🗓 Adding calendar features...")
    trend_df = add_calendar_features(trend_df, KEYWORDS)

    output_file = "google_trends_portugal_algarve_daily.csv"
    trend_df.to_csv(output_file, index=False)

    print("✅ DONE")
    print(trend_df.head())
    print(f"📁 Saved final dataset: {output_file}")


# ======================================================
# 6. RUN
# ======================================================
if __name__ == "__main__":
    main()
