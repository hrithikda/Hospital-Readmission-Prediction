import os
import json
import pandas as pd
from preprocess import load_and_clean, encode_features, get_features_and_target
from train import split_and_scale, train_xgboost, evaluate_model, cross_validate_model, save_model, save_scaler
from evaluate import compute_shap_values, save_shap_summary_plot, save_shap_bar_plot, get_feature_importance_df, binary_performance


DATA_PATH = 'data/cleaned_hospital_readmissions.csv'
MODELS_DIR = 'models'
ASSETS_DIR = 'assets'


def run():
    print('Loading and cleaning data...')
    df = load_and_clean(DATA_PATH)

    print('Encoding features...')
    df = encode_features(df)

    print('Splitting and scaling...')
    X, y = get_features_and_target(df)
    X_train, X_test, y_train, y_test, scaler = split_and_scale(X, y)

    print('Training XGBoost...')
    model = train_xgboost(X_train, y_train)

    print('Evaluating...')
    results = evaluate_model('XGBoost', model, X_test, y_test)
    print(f"  MSE:  {results['mse']}")
    print(f"  R²:   {results['r2']}")

    print('Running cross-validation...')
    cv_results = cross_validate_model(model, X_train, y_train, cv=5)
    print(f"  CV R² mean: {cv_results['cv_r2_mean']}")
    print(f"  CV R² std:  {cv_results['cv_r2_std']}")

    print('Computing SHAP values...')
    explainer, shap_values = compute_shap_values(model, X_test)

    print('Saving SHAP plots...')
    os.makedirs(ASSETS_DIR, exist_ok=True)
    save_shap_summary_plot(shap_values, X_test,
                           f'{ASSETS_DIR}/shap_summary.png')
    save_shap_bar_plot(shap_values, X_test,
                       f'{ASSETS_DIR}/shap_bar.png')

    importance_df = get_feature_importance_df(shap_values, list(X.columns))
    importance_df.to_csv(f'{ASSETS_DIR}/feature_importance.csv', index=False)

    y_pred = model.predict(X_test)
    binary = binary_performance(y_test.values, y_pred)

    print('Saving model and scaler...')
    os.makedirs(MODELS_DIR, exist_ok=True)
    save_model(model, f'{MODELS_DIR}/xgboost_model.pkl')
    save_scaler(scaler, f'{MODELS_DIR}/scaler.pkl')

    metrics = {**results, **cv_results, **binary}
    with open(f'{ASSETS_DIR}/metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)

    print('\nPipeline complete.')
    print(json.dumps(metrics, indent=2))


if __name__ == '__main__':
    run()