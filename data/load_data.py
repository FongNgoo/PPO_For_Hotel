# data/load_data.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline


# ======================================================
# LOAD HOTEL BOOKING DATA (for Logistic + PPO)
# ======================================================
def load_hotel_data(path, trends_path):  # Thêm trends_path
    df = pd.read_csv(path)
    df = df.dropna(subset=["children"])
    df = add_booking_date(df)
    trend_df = load_trends_data(trends_path)
    df = merge_trends(df, trend_df)
    
    numerical = [
        "lead_time", "stays_in_weekend_nights", "stays_in_week_nights",
        "adults", "children", "babies", "previous_cancellations",
        "trend_mean" # Thêm trends numerical
    ]
    categorical = [
        "market_segment", "distribution_channel", "arrival_date_month",
        "is_weekend", "is_holiday_pt", "is_before_holiday", "is_after_holiday"  # Treat as cat nếu cần, hoặc num
    ]
    # Pipeline cho numerical và categorical riêng
    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),  # hoặc 'median'
        ('scaler', StandardScaler())
    ])
    
    cat_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown="ignore"))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_pipeline, numerical),
            ("cat", cat_pipeline, categorical)
        ]
    )
    
    X = preprocessor.fit_transform(df)
    return X, df, preprocessor


# ======================================================
# LOAD GOOGLE TRENDS DATA
# Expected columns:
# - date
# - trend_index
# ======================================================
def load_trends_data(path):
    """
    Load background context data:
    - Google Trends (multiple keywords)
    - Holiday & calendar features
    """

    trends = pd.read_csv(path)

    # --------------------------------------------------
    # Build date column
    # --------------------------------------------------
    if "date" not in trends.columns:
        trends["date"] = pd.to_datetime(
            trends[["year", "month", "day"]]
        )
    else:
        trends["date"] = pd.to_datetime(trends["date"])

    # --------------------------------------------------
    # Ensure numeric types
    # --------------------------------------------------
    numeric_cols = [
        "resort algarve",
        "algarve hotel",
        "trend_mean",
        "is_holiday_pt",
        "is_weekend",
        "is_before_holiday",
        "is_after_holiday"
    ]

    for col in numeric_cols:
        if col in trends.columns:
            trends[col] = pd.to_numeric(trends[col], errors="coerce")

    trends = trends.sort_values("date")

    return trends

# ======================================================
# ADD BOOKING DATE (decision time)
# booking_date = arrival_date - lead_time
# ======================================================
def add_booking_date(df):
    """
    Robust creation of arrival_date and booking_date
    Handles mixed month formats: name / number
    """

    def parse_month(m):
        # numeric month
        if isinstance(m, (int, float)):
            return int(m)

        # string month
        m = str(m)

        # numeric string
        if m.isdigit():
            return int(m)

        # month name
        try:
            return pd.to_datetime(m, format="%B").month
        except:
            try:
                return pd.to_datetime(m, format="%b").month
            except:
                return np.nan

    # -----------------------------
    # Normalize month
    # -----------------------------
    df["arrival_month_num"] = df["arrival_date_month"].apply(parse_month)

    # -----------------------------
    # Arrival date
    # -----------------------------
    df["arrival_date"] = pd.to_datetime(
        dict(
            year=df["arrival_date_year"],
            month=df["arrival_month_num"],
            day=df["arrival_date_day_of_month"],
        ),
        errors="coerce"
    )

    # -----------------------------
    # Booking date
    # -----------------------------
    df["booking_date"] = df["arrival_date"] - pd.to_timedelta(
        df["lead_time"], unit="D"
    )

    return df

def merge_trends(hotel_df, trends_df):
    hotel_df['booking_date'] = pd.to_datetime(hotel_df['booking_date']).dt.date
    trends_df['date'] = pd.to_datetime(trends_df['date']).dt.date
    return hotel_df.merge(trends_df, left_on='booking_date', right_on='date', how='left')


# ======================================================
# BUILD BACKGROUND CONTEXT (shared by date)
# ======================================================
def build_background(date, trend_row):
    trend_cols = ["resort algarve", "algarve hotel"]

    trend_values = [
        float(trend_row[c]) for c in trend_cols if c in trend_row
    ]

    return {
        "is_weekend": bool(trend_row["is_weekend"]),
        "trend_mean": float(np.mean(trend_values)),
        "trend_max": float(np.max(trend_values)),
        "is_holiday_pt": bool(trend_row["is_holiday_pt"]),
        "is_before_holiday": bool(trend_row["is_before_holiday"]),
        "is_after_holiday": bool(trend_row["is_after_holiday"]),
    }



# ======================================================
# BUILD FINAL DATA STRUCTURE
# {date: {background, bookings}}
# ======================================================
def data_structure(hotel_df, trends_df):
    hotel_df = add_booking_date(hotel_df)

    data = {}

    for idx, row in hotel_df.iterrows():
        date = row["booking_date"].date()

        # Create new date node if needed
        if date not in data:
            trend_row = trends_df.loc[
                trends_df["date"] == pd.to_datetime(date)
            ]

            if trend_row.empty:
                continue

            trend_row = trend_row.iloc[0]

            data[date] = {
                "background": build_background(date, trend_row),
                "bookings": {}
            }

        booking_id = f"booking_{idx}"

        data[date]["bookings"][booking_id] = {
            # ----------------------------
            # Booking context
            # ----------------------------
            "booking_context": {
                "hotel_type": row["hotel"],
                "lead_time": int(row["lead_time"]),
                "arrival_date": str(row["arrival_date"].date()),
                "stays": {
                    "week_nights": int(row["stays_in_week_nights"]),
                    "weekend_nights": int(row["stays_in_weekend_nights"]),
                    "total_nights": int(
                        row["stays_in_week_nights"]
                        + row["stays_in_weekend_nights"]
                    )
                }
            },

            # ----------------------------
            # Customer profile
            # ----------------------------
            "customer_profile": {
                "adults": int(row["adults"]),
                "children": int(row["children"]),
                "babies": int(row["babies"]),
                "country": row["country"],
                "is_repeated_guest": int(row["is_repeated_guest"]),
                "previous_cancellations": int(row["previous_cancellations"]),
                "previous_bookings_not_canceled": int(
                    row["previous_bookings_not_canceled"]
                )
            },

            # ----------------------------
            # Channel information
            # ----------------------------
            "channel_info": {
                "market_segment": row["market_segment"],
                "distribution_channel": row["distribution_channel"]
            },

            # ----------------------------
            # Purchase intent signals
            # ----------------------------
            "intent_signals": {
                "booking_changes": int(row["booking_changes"]),
                "deposit_type": row["deposit_type"],
                "special_requests": int(row["total_of_special_requests"])
            },

            # ----------------------------
            # Pricing info
            # ----------------------------
            "pricing": {
                "observed_adr": float(row["adr"])
            },

            # ----------------------------
            # Label (ONLY for Logistic / evaluation)
            # ----------------------------
            "label": {
                "is_canceled": int(row["is_canceled"]),
                "accept": 1 - int(row["is_canceled"])
            }
        }

    return data
