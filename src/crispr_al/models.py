"""Model training for Design A."""
import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler


def scale_features(X_train: np.ndarray, X_test: np.ndarray) -> tuple:
    """Fit StandardScaler on X_train, transform both X_train and X_test."""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled


def train_ridge(
    X_train: np.ndarray,
    y_train: np.ndarray,
    alphas: list = None,
    cv: int = 5,
) -> RidgeCV:
    """Train RidgeCV regressor."""
    if alphas is None:
        alphas = [0.1, 1.0, 10.0, 100.0]
    model = RidgeCV(alphas=alphas, cv=cv)
    model.fit(X_train, y_train)
    return model


def train_rf(
    X_train: np.ndarray,
    y_train: np.ndarray,
    seed: int,
    n_estimators: int = 200,
    min_samples_leaf: int = 1,
) -> RandomForestRegressor:
    """Train RandomForestRegressor."""
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_features="sqrt",
        min_samples_leaf=min_samples_leaf,
        random_state=seed,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model


def predict(model, X_test: np.ndarray) -> np.ndarray:
    """Generate predictions from a fitted model."""
    return model.predict(X_test)
