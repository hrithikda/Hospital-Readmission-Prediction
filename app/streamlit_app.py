import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import shap
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

st.set_page_config(
    page_title='Hospital Readmission Risk Predictor',
    page_icon='🏥',
    layout='wide'
)

MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'xgboost_model.pkl')
SCALER_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'scaler.pkl')
METRICS_PATH = os.path.join(os.path.dirname(__file__), '..', 'assets', 'metrics.json')
IMPORTANCE_PATH = os.path.join(os.path.dirname(__file__), '..', 'assets', 'feature_importance.csv')
SHAP_BAR_PATH = os.path.join(os.path.dirname(__file__), '..', 'assets', 'shap_bar.png')
SHAP_SUMMARY_PATH = os.path.join(os.path.dirname(__file__), '..', 'assets', 'shap_summary.png')


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_resource
def load_scaler():
    return joblib.load(SCALER_PATH)


@st.cache_data
def load_metrics():
    with open(METRICS_PATH) as f:
        return json.load(f)


@st.cache_data
def load_importance():
    return pd.read_csv(IMPORTANCE_PATH)


model = load_model()
scaler = load_scaler()
metrics = load_metrics()
importance_df = load_importance()

st.title('🏥 Hospital Readmission Risk Predictor')
st.markdown(
    'Predicting **Excess Readmission Ratios** for US hospitals using the '
    'FY2024 CMS Hospital Readmissions Reduction Program (HRRP) dataset. '
    'A ratio above 1.0 means the hospital has higher-than-expected readmissions '
    'and faces CMS financial penalties.'
)

st.markdown('---')

tab1, tab2, tab3 = st.tabs(['Predict', 'Model Performance', 'Feature Importance'])

with tab1:
    st.subheader('Hospital Risk Assessment')
    st.markdown('Enter hospital metrics below to predict the Excess Readmission Ratio.')

    col1, col2 = st.columns(2)

    with col1:
        num_discharges = st.number_input(
            'Number of Discharges',
            min_value=1, max_value=50000, value=500,
            help='Total patient discharges for this measure period'
        )
        predicted_rate = st.number_input(
            'Predicted Readmission Rate (%)',
            min_value=0.0, max_value=50.0, value=15.0, step=0.1,
            help='Hospital predicted readmission rate based on patient mix'
        )
        expected_rate = st.number_input(
            'Expected Readmission Rate (%)',
            min_value=0.0, max_value=50.0, value=15.0, step=0.1,
            help='National benchmark readmission rate for this condition'
        )

    with col2:
        num_readmissions = st.number_input(
            'Number of Readmissions',
            min_value=0, max_value=10000, value=75,
            help='Actual number of readmissions observed'
        )
        state_enc = st.slider(
            'State (encoded)',
            min_value=0, max_value=50, value=14,
            help='Numeric encoding of the hospital state (0=AL to 50=WY)'
        )
        measure = st.selectbox(
            'Condition Measure',
            options=[
                'AMI (Heart Attack)',
                'CABG (Heart Bypass)',
                'COPD',
                'Heart Failure',
                'Hip/Knee Replacement',
                'Pneumonia'
            ]
        )

    measure_map = {
        'AMI (Heart Attack)':      [False, False, False, False, False],
        'CABG (Heart Bypass)':     [True,  False, False, False, False],
        'COPD':                    [False, True,  False, False, False],
        'Heart Failure':           [False, False, True,  False, False],
        'Hip/Knee Replacement':    [False, False, False, True,  False],
        'Pneumonia':               [False, False, False, False, True],
    }
    measure_flags = measure_map[measure]

    facility_enc = predicted_rate / (expected_rate + 1e-9)

    feature_values = np.array([[
        num_discharges,
        predicted_rate,
        expected_rate,
        num_readmissions,
        999,
        999,
        facility_enc,
        state_enc,
        *measure_flags
    ]])

    FEATURE_NAMES = [
        'Number of Discharges', 'Predicted Readmission Rate',
        'Expected Readmission Rate', 'Number of Readmissions',
        'Start Year', 'End Year', 'Facility_Name_Enc', 'State_Enc',
        'Measure Name_READM-30-CABG-HRRP', 'Measure Name_READM-30-COPD-HRRP',
        'Measure Name_READM-30-HF-HRRP', 'Measure Name_READM-30-HIP-KNEE-HRRP',
        'Measure Name_READM-30-PN-HRRP'
    ]

    input_df = pd.DataFrame(feature_values, columns=FEATURE_NAMES)
    input_scaled = scaler.transform(input_df)

    if st.button('Predict Readmission Risk', type='primary'):
        prediction = model.predict(input_scaled)[0]

        st.markdown('---')
        col_r1, col_r2, col_r3 = st.columns(3)

        with col_r1:
            st.metric(
                label='Excess Readmission Ratio',
                value=f'{prediction:.4f}',
                delta=f'{prediction - 1.0:+.4f} vs benchmark'
            )

        with col_r2:
            risk_label = 'HIGH RISK' if prediction > 1.0 else 'LOW RISK'
            risk_color = '🔴' if prediction > 1.0 else '🟢'
            st.metric(label='CMS Penalty Risk', value=f'{risk_color} {risk_label}')

        with col_r3:
            pct_above = ((prediction - 1.0) / 1.0) * 100
            st.metric(
                label='Distance from Benchmark',
                value=f'{abs(pct_above):.2f}%',
                delta='above benchmark' if prediction > 1.0 else 'below benchmark'
            )

        if prediction > 1.0:
            st.error(
                f'This hospital is predicted to have **{pct_above:.1f}% excess readmissions** '
                f'above the national benchmark. CMS financial penalties may apply under the HRRP.'
            )
        else:
            st.success(
                f'This hospital is performing **{abs(pct_above):.1f}% below** the national '
                f'readmission benchmark. No CMS penalty risk detected.'
            )

        st.markdown('#### SHAP Explanation for this Prediction')
        explainer = shap.Explainer(model)
        shap_vals = explainer(input_df)
        fig, ax = plt.subplots(figsize=(10, 4))
        shap.plots.waterfall(shap_vals[0], show=False)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

with tab2:
    st.subheader('Model Performance')
    st.markdown(
        'XGBoost regressor trained on 18,774 hospital-condition records '
        'from the FY2024 CMS HRRP dataset.'
    )

    col1, col2, col3, col4 = st.columns(4)
    col1.metric('R² Score', f"{metrics['r2']:.4f}")
    col2.metric('CV R² Mean', f"{metrics['cv_r2_mean']:.4f}")
    col3.metric('CV R² Std', f"± {metrics['cv_r2_std']:.4f}")
    col4.metric('MSE', f"{metrics['mse']:.5f}")

    st.markdown('#### Binary Classification (Excess vs Non-Excess)')
    st.markdown(
        'Hospitals with predicted ratio > 1.0 are classified as excess readmission risk.'
    )

    col5, col6, col7 = st.columns(3)
    col5.metric('Precision', f"{metrics['precision']:.4f}")
    col6.metric('Recall', f"{metrics['recall']:.4f}")
    col7.metric('F1 Score', f"{metrics['f1']:.4f}")

    st.markdown('#### SHAP Summary Plot')
    st.image(SHAP_SUMMARY_PATH, use_container_width=True)

with tab3:
    st.subheader('Feature Importance (SHAP)')
    st.markdown(
        'Mean absolute SHAP values across the test set. '
        'Higher values mean the feature has more influence on the predicted ratio.'
    )

    top_features = importance_df.head(10).copy()
    top_features.columns = ['Feature', 'Mean |SHAP|']
    top_features['Mean |SHAP|'] = top_features['Mean |SHAP|'].round(5)

    st.dataframe(top_features, use_container_width=True, hide_index=True)

    st.markdown('#### SHAP Bar Chart')
    st.image(SHAP_BAR_PATH, use_container_width=True)