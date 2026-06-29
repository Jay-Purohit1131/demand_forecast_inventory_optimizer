import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import joblib
import pytest

MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'lgbm_final.joblib')


@pytest.mark.skipif(not os.path.exists(MODEL_PATH), reason="model not trained yet")
def test_model_predicts_sane_output():
    """Load the saved model, run a few rows, assert output is sane."""
    model = joblib.load(MODEL_PATH)
    n_features = model.n_features_in_
    # one synthetic row of the right width
    X = np.zeros((3, n_features))
    preds = model.predict(X)

    assert preds.shape == (3,), "wrong output shape"
    assert np.all(np.isfinite(preds)), "predictions contain NaN/inf"
    # sales forecasts should be clip-able to non-negative; raw can be slightly
    # negative, so we just check they're not wildly broken
    assert np.all(preds < 1e6), "predictions implausibly large"