import numpy as np

def seasonal_naive_scale(train_df, season=7):
    """Mean |y_t - y_{t-season}| computed WITHIN each series, then pooled."""
    diffs = []
    for _, g in train_df.groupby('id'):
        v = g.sort_values('date')['sales'].values
        if len(v) > season:
            diffs.append(np.abs(v[season:] - v[:-season]))
    return np.concatenate(diffs).mean()


def mase(y_true, y_pred, scale):
    mae_model = np.mean(np.abs(y_true - y_pred))
    return mae_model / scale if scale != 0 else np.nan

def seasonal_naive_pred(df):
    """Seasonal-naive forecast = sales from 7 days earlier (lag_7)."""
    return df['lag_7']