import pandas as pd

def expanding_window_splits(df, horizon=28, nfolds =3, date_col = 'date'):
    """
    Yield (train_idx, test_idx) for an expanding window backtest.

    Splitting on DATE, so the same calendar cutoff applies
    across all series at once. Training set expands each fold.
    Each test window is 'horizon' days and never overlaps with training data.
    """

    unique_days = pd.Series(df[date_col].unique()).sort_values().reset_index(drop=True)
    last_day = unique_days.iloc[-1]

    for fold in range(nfolds):
        test_end = last_day - pd.Timedelta(days=fold * horizon)
        test_start = test_end - pd.Timedelta(days=horizon - 1)
        train_end = test_start - pd.Timedelta(days=1)

        train_idx = df.index[df[date_col] <= train_end]
        test_idx = df.index[(df[date_col] >= test_start) & (df[date_col] <= test_end)]

        yield train_idx, test_idx