import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
import pytest
from src.features import add_lag_rolling_features, add_calendar_price_feature


def make_series(n_days=60, n_series=2):
    """Build a tiny synthetic dataset with the columns the feature code needs."""
    rows = []
    for s in range(n_series):
        for d in range(1, n_days + 1):
            rows.append({
                'id': f'item_{s}',
                'date': pd.Timestamp('2014-01-01') + pd.Timedelta(days=d),
                'sales': np.random.randint(0, 5),
                'event_name_1': np.nan,
                'snap_CA': d % 2,
            })
    return pd.DataFrame(rows)


def test_no_future_leakage_in_features():
    """
    Features for an early date must not change when later days are removed.
    A leak-free (backward-only) feature cannot depend on the future, so
    truncating the future leaves past rows' features identical.
    """
    df_full = make_series(n_days=60)
    df_trunc = df_full[df_full['date'] <= '2014-02-15'].copy()  # chop off the tail

    feat_full = add_lag_rolling_features(df_full)
    feat_trunc = add_lag_rolling_features(df_trunc)

    feat_cols = [c for c in feat_full.columns if c.startswith(('lag_', 'rolling_'))]

    # Compare the overlapping early rows, keyed by id+date
    merged = feat_full.merge(
        feat_trunc, on=['id', 'date'], suffixes=('_full', '_trunc')
    )
    for c in feat_cols:
        a = merged[f'{c}_full']
        b = merged[f'{c}_trunc']
        # equal where both are non-NaN (NaNs are the warm-up period)
        mask = a.notna() & b.notna()
        assert np.allclose(a[mask], b[mask]), f"Leakage detected in feature: {c}"


def test_rolling_excludes_current_day():
    """rolling_mean_7 on a row must not include that row's own sales."""
    df = make_series(n_days=30, n_series=1)
    feat = add_lag_rolling_features(df)
    # For a row with a populated rolling_mean_7, recompute the mean of the
    # 7 PRIOR days' sales and confirm it matches (i.e. current day excluded).
    feat = feat.sort_values('date').reset_index(drop=True)
    row = feat[feat['rolling_mean_7'].notna()].iloc[0]
    pos = feat.index[feat['date'] == row['date']][0]
    prior_7 = feat['sales'].iloc[pos-7:pos].mean()
    assert np.isclose(row['rolling_mean_7'], prior_7), "rolling_mean_7 includes current day"


def test_calendar_features_present():
    """add_calendar_price_feature should produce the expected columns."""
    df = make_series(n_days=10)
    df['date'] = pd.to_datetime(df['date'])
    out = add_calendar_price_feature(df)
    for col in ['is_event', 'snap', 'day']:
        assert col in out.columns