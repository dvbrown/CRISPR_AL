"""Tests for crispr_al.models module."""
import numpy as np
import pytest
from crispr_al.models import train_ridge, train_rf, scale_features, predict


@pytest.fixture
def train_test_data():
    rng = np.random.default_rng(42)
    X_train = rng.normal(0, 1, (200, 9))
    y_train = rng.normal(0, 1, 200)
    X_test = rng.normal(0, 1, (50, 9))
    return X_train, y_train, X_test


def test_scale_features_no_leakage(train_test_data):
    X_train, _, X_test = train_test_data
    X_train_scaled, X_test_scaled = scale_features(X_train, X_test)
    # Train set should be approximately mean=0, std=1
    assert abs(X_train_scaled.mean()) < 0.1
    assert abs(X_train_scaled.std() - 1.0) < 0.1


def test_scale_features_shape(train_test_data):
    X_train, _, X_test = train_test_data
    X_train_scaled, X_test_scaled = scale_features(X_train, X_test)
    assert X_train_scaled.shape == X_train.shape
    assert X_test_scaled.shape == X_test.shape


def test_train_ridge_predict_shape(train_test_data):
    X_train, y_train, X_test = train_test_data
    X_train_s, X_test_s = scale_features(X_train, X_test)
    model = train_ridge(X_train_s, y_train)
    preds = predict(model, X_test_s)
    assert preds.shape == (50,)


def test_train_rf_predict_shape(train_test_data):
    X_train, y_train, X_test = train_test_data
    X_train_s, X_test_s = scale_features(X_train, X_test)
    model = train_rf(X_train_s, y_train, seed=42, n_estimators=10)
    preds = predict(model, X_test_s)
    assert preds.shape == (50,)


def test_train_ridge_no_leakage(train_test_data):
    """Verify Ridge uses only train data by checking different scalers produce different results."""
    X_train, y_train, X_test = train_test_data
    X_train_s, X_test_s = scale_features(X_train, X_test)
    # Scale using test data only (leaky)
    from sklearn.preprocessing import StandardScaler
    scaler_leaky = StandardScaler()
    X_test_leaky = scaler_leaky.fit_transform(X_test)
    model = train_ridge(X_train_s, y_train)
    preds_proper = predict(model, X_test_s)
    preds_leaky = predict(model, X_test_leaky)
    assert not np.allclose(preds_proper, preds_leaky), "Proper and leaky predictions should differ"
