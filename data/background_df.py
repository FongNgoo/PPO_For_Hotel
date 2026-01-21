"""
Google Trends + Portugal Holidays
Use case: Hotel / Tourism Demand Forecasting (Algarve, Portugal)
"""

from pytrends.request import TrendReq
import pandas as pd
import holidays


# ======================================================
# 1. CONFIG
# ======================================================
START_DATE = "2013-06-23"
END_DATE = "2017-09-01"

KEYWORDS = [
    "resort algarve",
    "algarve hotel"
]

GEO = "PT"


# ======================================================
# 2. GOOGLE TRENDS
# ======================================================
def get_google_trends(keywords, geo, start_date, end_date):
    pytrends = TrendReq(
        hl="en-US",
        tz=0
    )

    timeframe = f"{start_date} {end_date}"

    pytrends.build_payload(
        kw_list=keywords,
        geo=geo,
        timeframe=timeframe
    )

    df = pytrends.interest_over_time()

    if df.empty:
        raise ValueError("Google Trends returned empty dataframe")

    df = df.reset_index()

    if "isPartial" in df.columns:
        df = df.drop(columns=["isPartial"])

    return df


# ======================================================
# 3. PORTUGAL HOLIDAYS
# ======================================================
def add_portugal_holidays(df):
    years = range(df["date"].dt.year.min(),
                  df["date"].dt.year.max() + 1)

    pt_holidays = holidays.Portugal(years=years)

    df["is_holiday_pt"] = df["date"].isin(pt_holidays)
    df["holiday_name"] = df["date"].map(pt_holidays)

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

    # Mean trend (recommended)
    df["trend_mean"] = df[keywords].mean(axis=1)

    # Holiday effect
    df["is_before_holiday"] = df["date"].shift(-1).isin(
        df.loc[df["is_holiday_pt"], "date"]
    )
    df["is_after_holiday"] = df["date"].shift(1).isin(
        df.loc[df["is_holiday_pt"], "date"]
    )

    return df


# ======================================================
# 5. MAIN PIPELINE
# ======================================================
def main():
    print("Downloading Google Trends data...")
    trend_df = get_google_trends(
        KEYWORDS, GEO, START_DATE, END_DATE
    )

    print("Adding Portugal holidays...")
    trend_df = add_portugal_holidays(trend_df)

    print("Adding calendar features...")
    trend_df = add_calendar_features(trend_df, KEYWORDS)

    print("Done!")
    print(trend_df.head())

    # Save
    trend_df.to_csv(
        "google_trends_portugal_algarve.csv",
        index=False
    )
    print("Saved: google_trends_portugal_algarve.csv")


# ======================================================
# 6. RUN
# ======================================================
if __name__ == "__main__":
    main()
