import pandas as pd
import numpy as np

def add_lag_rolling_features(df, lags=(7, 14, 28), windows=(7, 28)):
    """..."""  # keep your docstring
    df = df.sort_values(['id', 'date']).reset_index(drop=True)
    g = df.groupby('id')['sales']

    for lag in lags:
        df[f'lag_{lag}'] = g.shift(lag)

    shifted = df.groupby('id')['sales'].shift(1)
    for w in windows:
        df[f'rolling_mean_{w}'] = (
            shifted.groupby(df['id']).rolling(w).mean()
            .reset_index(level=0, drop=True)
        )
        df[f'rolling_std_{w}'] = (
            shifted.groupby(df['id']).rolling(w).std()
            .reset_index(level=0, drop=True)
        )
    return df


def add_calendar_price_feature(df):
    """Shape calendar, event, snap, and price columns into model-ready form."""
    df = df.copy()
    df['is_event'] = df['event_name_1'].notna().astype(int)
    df['snap'] = df['snap_CA'].notna().astype(int)
    df['day'] = df['date'].dt.day
    return df