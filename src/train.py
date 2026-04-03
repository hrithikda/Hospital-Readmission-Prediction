import joblib
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor


def split_and_scale(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler


def train_xgboost(X_train, y_train):
    model = XGBRegressor(
        objective='reg:squarederror',
        n_estimators=200,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
    model.fit(X_train, y_train)
    return model


def evaluate_model(name, model, X_test, y_test):
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return {'model': name, 'mse': round(mse, 5), 'r2': round(r2, 5)}


def cross_validate_model(model, X_train, y_train, cv=5):
    scores = cross_val_score(
        model, X_train, y_train, cv=cv, scoring='r2'
    )
    return {
        'cv_r2_mean': round(scores.mean(), 4),
        'cv_r2_std': round(scores.std(), 4)
    }


def save_model(model, path):
    joblib.dump(model, path)


def save_scaler(scaler, path):
    joblib.dump(scaler, path)