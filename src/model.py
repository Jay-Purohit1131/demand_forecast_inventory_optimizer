import lightgbm as lgb
import numpy as np

def score_params(df, params, features, target, cat_features,
                 splits_fn, scale_fn):
    """Mean MASE across time-aware folds for one param set."""
    fold_scores = []
    for train_idx, test_idx in splits_fn(df):
        tr, te = df.loc[train_idx], df.loc[test_idx]
        model = lgb.LGBMRegressor(**params, n_jobs=-1, verbose=-1)
        model.fit(tr[features], tr[target], categorical_feature=cat_features)
        pred = np.clip(model.predict(te[features]), 0, None)
        scale = scale_fn(tr)
        fold_scores.append(np.mean(np.abs(te[target].values - pred)) / scale)
    return np.mean(fold_scores)

def train_quantile_models(df, features, target, cat_features,
                          quantiles=(0.1, 0.5, 0.9), params=None):
    """Train one LightGBM per quantile. Returns {quantile: fitted_model}."""
    params = params or {'n_estimators': 500, 'learning_rate': 0.05, 'num_leaves': 63}
    models = {}
    for q in quantiles:
        m = lgb.LGBMRegressor(objective='quantile', alpha=q,
                              **params, n_jobs=-1, verbose=-1)
        m.fit(df[features], df[target], categorical_feature=cat_features)
        models[q] = m
    return models