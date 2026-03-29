# data/load_data.py
"""
Data loading and preprocessing for Multi-Room Hotel Pricing.

This module handles:
1. Loading hotel booking data and Google Trends
2. Building daily-level features for each room type
3. Creating train/test splits (temporal)
4. Preprocessing for demand models and RL environment

UPDATED: Auto-detect room types and ADR from data instead of hardcoding
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


# Minimum bookings required for a room type to be included
MIN_BOOKINGS_PER_ROOM = 50


def detect_room_types_and_adr(
    hotel_df: pd.DataFrame,
    min_bookings: int = MIN_BOOKINGS_PER_ROOM
) -> Tuple[List[str], Dict[str, float], Dict[str, int]]:
    """
    Auto-detect room types with sufficient data and compute reference ADR.
    
    Args:
        hotel_df: Hotel booking DataFrame
        min_bookings: Minimum bookings required to include a room type
        
    Returns:
        room_types: List of room types with sufficient data
        adr_refs: Dict mapping room_type -> mean ADR
    """
    # Count bookings per room type
    room_counts = hotel_df['reserved_room_type'].value_counts()
    
    # Filter to rooms with enough bookings
    valid_rooms = room_counts[room_counts >= min_bookings].index.tolist()
    
    # Sort by booking count (descending)
    valid_rooms = sorted(valid_rooms, key=lambda r: room_counts[r], reverse=True)
    
    # Compute mean ADR and max historical capacity for each room type
    adr_refs = {}
    room_capacities = {}
    for room in valid_rooms:
        room_df = hotel_df[hotel_df['reserved_room_type'] == room]
        adr_refs[room] = room_df['adr'].mean()
        max_daily = room_df.groupby('arrival_date').size().max()
        room_capacities[room] = int(np.ceil(max_daily * 1.05))
    
    print(f"Detected {len(valid_rooms)} room types with >= {min_bookings} bookings:")
    for room in valid_rooms:
        print(f"  Room {room}: {room_counts[room]} bookings, ADR=€{adr_refs[room]:.0f}, Capacity={room_capacities[room]}")
    
    return valid_rooms, adr_refs, room_capacities


def load_raw_data(
    hotel_path: str,
    trends_path: str,
    min_bookings: int = MIN_BOOKINGS_PER_ROOM
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str], Dict[str, float], Dict[str, int]]:
    """Load raw hotel and trends data, auto-detect room types."""
    
    hotel_df = pd.read_csv(hotel_path)
    trends_df = pd.read_csv(trends_path)
    
    # Parse dates
    hotel_df['arrival_date'] = pd.to_datetime(
        hotel_df['arrival_date_year'].astype(str) + '-' +
        hotel_df['arrival_date_month'].astype(str) + '-' +
        hotel_df['arrival_date_day_of_month'].astype(str),
        format='mixed'
    )
    
    # Remove canceled bookings for demand modeling (we model actual bookings)
    hotel_df = hotel_df[hotel_df['is_canceled'] == 0].copy()
    
    # Auto-detect room types, ADR, and capacity references
    room_types, adr_refs, room_capacities = detect_room_types_and_adr(hotel_df, min_bookings)
    
    # Filter to included room types
    hotel_df = hotel_df[hotel_df['reserved_room_type'].isin(room_types)].copy()
    
    # Parse trends date
    trends_df['date'] = pd.to_datetime(trends_df['date'])
    
    print(f"\nLoaded {len(hotel_df)} bookings across {hotel_df['arrival_date'].nunique()} dates")
    print(f"Room types: {room_types}")
    
    return hotel_df, trends_df, room_types, adr_refs, room_capacities


def build_daily_dataset(
    hotel_df: pd.DataFrame,
    trends_df: pd.DataFrame,
    room_types: List[str],
    adr_refs: Dict[str, float]
) -> pd.DataFrame:
    """
    Build daily-level dataset with features for each date.
    
    Returns DataFrame indexed by date with columns:
    - Temporal features (day_of_week, month, etc.)
    - Trend features (google_trends, momentum)
    - Per-room features (bookings_7d, avg_adr_7d, etc.)
    """
    
    # Get date range
    dates = hotel_df['arrival_date'].sort_values().unique()
    min_date, max_date = dates[0], dates[-1]
    
    # Create full date range
    date_range = pd.date_range(min_date, max_date, freq='D')
    daily_df = pd.DataFrame({'date': date_range})
    
    # ========================================
    # 1. TEMPORAL FEATURES
    # ========================================
    daily_df['day_of_week'] = daily_df['date'].dt.dayofweek
    daily_df['month'] = daily_df['date'].dt.month
    daily_df['is_weekend'] = daily_df['day_of_week'].isin([5, 6]).astype(int)
    daily_df['day_of_year'] = daily_df['date'].dt.dayofyear
    
    # Holidays (Portuguese holidays - simplified)
    portuguese_holidays = [
        '01-01', '04-25', '05-01', '06-10', '08-15', 
        '10-05', '11-01', '12-01', '12-08', '12-25'
    ]
    daily_df['date_str'] = daily_df['date'].dt.strftime('%m-%d')
    daily_df['is_holiday'] = daily_df['date_str'].isin(portuguese_holidays).astype(int)
    daily_df.drop('date_str', axis=1, inplace=True)
    
    # ========================================
    # 2. GOOGLE TRENDS
    # ========================================
    # Get trend column name
    trend_col = [c for c in trends_df.columns if 'algarve' in c.lower() or 'trend' in c.lower()]
    if trend_col:
        trend_col = trend_col[0]
    else:
        trend_col = trends_df.columns[1]  # Second column
    
    # Rename date column if needed
    date_col = trends_df.columns[0]  # First column is usually date
    trends_df = trends_df.rename(columns={
        date_col: 'trend_date',
        trend_col: 'google_trends'
    })
    trends_df['trend_date'] = pd.to_datetime(trends_df['trend_date'])
    
    # Keep only needed columns
    trends_df = trends_df[['trend_date', 'google_trends']]
    
    # Merge trends
    daily_df = daily_df.merge(
        trends_df, 
        left_on='date', 
        right_on='trend_date', 
        how='left'
    )
    
    # Drop trend_date column if it exists
    if 'trend_date' in daily_df.columns:
        daily_df.drop('trend_date', axis=1, inplace=True)
    
    # Fill missing trends with median
    daily_df['google_trends'] = daily_df['google_trends'].fillna(
        daily_df['google_trends'].median()
    )
    
    # Normalize trends to [0, 1]
    max_trend = daily_df['google_trends'].max()
    if max_trend > 0:
        daily_df['google_trends'] = daily_df['google_trends'] / max_trend
    
    # Trend momentum (7-day change)
    daily_df['trend_momentum'] = daily_df['google_trends'].pct_change(7).fillna(0)
    daily_df['trend_momentum'] = daily_df['trend_momentum'].clip(-1, 1)
    
    # ========================================
    # 3. PER-ROOM HISTORICAL FEATURES
    # ========================================
    # Aggregate bookings by date and room type
    daily_room_stats = hotel_df.groupby(['arrival_date', 'reserved_room_type']).agg({
        'adr': ['count', 'mean'],
        'lead_time': 'mean',
        'stays_in_weekend_nights': 'mean',
        'stays_in_week_nights': 'mean'
    }).reset_index()
    
    daily_room_stats.columns = [
        'date', 'room_type', 'bookings', 'avg_adr', 
        'avg_lead_time', 'avg_weekend_nights', 'avg_week_nights'
    ]
    
    # Pivot to get columns per room type
    for room in room_types:
        room_data = daily_room_stats[daily_room_stats['room_type'] == room].copy()
        room_data = room_data.set_index('date')
        
        # Reindex to full date range, fill with 0
        room_data = room_data.reindex(date_range)
        room_data['bookings'] = room_data['bookings'].fillna(0)
        room_data['avg_adr'] = room_data['avg_adr'].fillna(adr_refs[room])
        room_data['avg_lead_time'] = room_data['avg_lead_time'].fillna(
            room_data['avg_lead_time'].median() if room_data['avg_lead_time'].notna().any() else 30
        )
        
        # Rolling features (7-day)
        daily_df[f'bookings_7d_{room}'] = room_data['bookings'].rolling(7, min_periods=1).mean().values
        daily_df[f'avg_adr_7d_{room}'] = room_data['avg_adr'].rolling(7, min_periods=1).mean().values
        daily_df[f'bookings_yesterday_{room}'] = room_data['bookings'].shift(1).fillna(0).values
        
        # Normalized features
        max_bookings = daily_df[f'bookings_7d_{room}'].max()
        if max_bookings > 0:
            daily_df[f'bookings_7d_{room}'] = daily_df[f'bookings_7d_{room}'] / max_bookings
        daily_df[f'avg_adr_7d_{room}'] = daily_df[f'avg_adr_7d_{room}'] / adr_refs[room]
    
    # ========================================
    # 4. GLOBAL FEATURES
    # ========================================
    total_daily = hotel_df.groupby('arrival_date').agg({
        'adr': 'count',
        'lead_time': 'mean'
    }).rename(columns={'adr': 'total_bookings', 'lead_time': 'avg_lead_time'})
    
    total_daily = total_daily.reindex(date_range).fillna(0)
    
    daily_df['total_bookings_yesterday'] = total_daily['total_bookings'].shift(1).fillna(0).values
    daily_df['total_bookings_7d'] = total_daily['total_bookings'].rolling(7, min_periods=1).mean().values
    
    # Normalize
    max_bookings_yesterday = daily_df['total_bookings_yesterday'].max()
    max_bookings_7d = daily_df['total_bookings_7d'].max()
    if max_bookings_yesterday > 0:
        daily_df['total_bookings_yesterday'] = daily_df['total_bookings_yesterday'] / max_bookings_yesterday
    if max_bookings_7d > 0:
        daily_df['total_bookings_7d'] = daily_df['total_bookings_7d'] / max_bookings_7d
    
    # ========================================
    # 5. TARGET: ACTUAL BOOKINGS PER ROOM (for training)
    # ========================================
    for room in room_types:
        room_bookings = hotel_df[hotel_df['reserved_room_type'] == room].groupby('arrival_date').size()
        room_bookings = room_bookings.reindex(date_range).fillna(0)
        daily_df[f'target_bookings_{room}'] = room_bookings.values
        
        # Also store actual ADR for that day
        room_adr = hotel_df[hotel_df['reserved_room_type'] == room].groupby('arrival_date')['adr'].mean()
        room_adr = room_adr.reindex(date_range).fillna(adr_refs[room])
        daily_df[f'target_adr_{room}'] = room_adr.values
    
    # Drop first 7 days (need history for rolling features)
    daily_df = daily_df.iloc[7:].reset_index(drop=True)
    
    print(f"Built daily dataset: {len(daily_df)} days, {len(daily_df.columns)} features")
    
    return daily_df


def create_temporal_split(
    daily_df: pd.DataFrame,
    test_ratio: float = 0.2
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split data temporally: earlier dates for training, later for testing.
    """
    n = len(daily_df)
    split_idx = int(n * (1 - test_ratio))
    
    train_df = daily_df.iloc[:split_idx].copy()
    test_df = daily_df.iloc[split_idx:].copy()
    
    print(f"Train: {len(train_df)} days ({train_df['date'].min()} to {train_df['date'].max()})")
    print(f"Test:  {len(test_df)} days ({test_df['date'].min()} to {test_df['date'].max()})")
    
    return train_df, test_df


def get_feature_columns(room_types: List[str]) -> Tuple[List[str], List[str]]:
    """Get lists of numerical and categorical feature columns."""
    
    numerical_features = [
        'is_weekend', 'is_holiday', 'day_of_year',
        'google_trends', 'trend_momentum',
        'total_bookings_yesterday', 'total_bookings_7d'
    ]
    
    # Per-room features
    for room in room_types:
        numerical_features.extend([
            f'bookings_7d_{room}',
            f'avg_adr_7d_{room}',
            f'bookings_yesterday_{room}'
        ])
    
    categorical_features = ['day_of_week', 'month']
    
    return numerical_features, categorical_features


def create_preprocessor(
    numerical_features: List[str],
    categorical_features: List[str]
) -> ColumnTransformer:
    """Create sklearn preprocessor for state encoding."""
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(drop='first', sparse_output=False), categorical_features)
        ],
        remainder='drop'
    )
    
    return preprocessor


def load_and_prepare_all(
    hotel_path: str = 'data/resort_hotel_data.csv',
    trends_path: str = 'data/google_trends_portugal_algarve_daily.csv',
    test_ratio: float = 0.2,
    min_bookings: int = MIN_BOOKINGS_PER_ROOM
) -> Dict:
    """
    Main function to load and prepare all data.
    
    Returns dict with:
    - hotel_df: raw hotel data
    - daily_df: daily features
    - train_df, test_df: temporal split
    - preprocessor: sklearn preprocessor
    - feature_columns: list of feature names
    - room_types: auto-detected room types
    - adr_refs: auto-computed reference ADRs
    """
    
    # Load raw data with auto-detected room types
    hotel_df, trends_df, room_types, adr_refs, room_capacities = load_raw_data(
        hotel_path, trends_path, min_bookings
    )
    
    # Build daily dataset
    daily_df = build_daily_dataset(hotel_df, trends_df, room_types, adr_refs)
    
    # Temporal split
    train_df, test_df = create_temporal_split(daily_df, test_ratio)
    
    # Get feature columns
    numerical_features, categorical_features = get_feature_columns(room_types)
    
    # Create preprocessor
    preprocessor = create_preprocessor(numerical_features, categorical_features)
    
    # Fit preprocessor on training data
    preprocessor.fit(train_df[numerical_features + categorical_features])
    
    return {
        'hotel_df': hotel_df,
        'trends_df': trends_df,
        'daily_df': daily_df,
        'train_df': train_df,
        'test_df': test_df,
        'preprocessor': preprocessor,
        'numerical_features': numerical_features,
        'categorical_features': categorical_features,
        'room_types': room_types,
        'adr_refs': adr_refs,
        'room_capacities': room_capacities
    }


if __name__ == '__main__':
    # Test loading
    data = load_and_prepare_all()
    
    print("\n" + "=" * 50)
    print("Data loading test complete!")
    print(f"Daily features shape: {data['daily_df'].shape}")
    print(f"Train: {len(data['train_df'])} days")
    print(f"Test: {len(data['test_df'])} days")
    print(f"Room types: {data['room_types']}")
    print(f"ADR refs: {data['adr_refs']}")
