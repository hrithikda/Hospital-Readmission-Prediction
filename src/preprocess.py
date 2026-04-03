import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.preprocessing import LabelEncoder
import category_encoders as ce


def load_and_clean(filepath):
    df = pd.read_csv(filepath)

    if 'Footnote' in df.columns:
        df.drop(columns=['Footnote'], inplace=True)

    df['Number of Readmissions'] = pd.to_numeric(
        df['Number of Readmissions'], errors='coerce'
    )

    num_cols = [
        'Number of Discharges',
        'Excess Readmission Ratio',
        'Predicted Readmission Rate',
        'Expected Readmission Rate',
        'Number of Readmissions'
    ]
    existing_num_cols = [c for c in num_cols if c in df.columns]
    imputer = KNNImputer(n_neighbors=5)
    df[existing_num_cols] = imputer.fit_transform(df[existing_num_cols])

    df.dropna(subset=['Excess Readmission Ratio'], inplace=True)

    if 'Start Date' in df.columns:
        df['Start Year'] = pd.to_datetime(df['Start Date']).dt.year
        df.drop(columns=['Start Date'], inplace=True)
    if 'End Date' in df.columns:
        df['End Year'] = pd.to_datetime(df['End Date']).dt.year
        df.drop(columns=['End Date'], inplace=True)

    return df


def encode_features(df):
    target_col = 'Excess Readmission Ratio'

    if 'Facility Name' in df.columns:
        target_enc = ce.TargetEncoder(cols=['Facility Name'])
        df['Facility_Name_Enc'] = target_enc.fit_transform(
            df['Facility Name'], df[target_col]
        )
        df.drop(columns=['Facility Name'], inplace=True)

    if 'State' in df.columns:
        le = LabelEncoder()
        df['State_Enc'] = le.fit_transform(df['State'])
        df.drop(columns=['State'], inplace=True)

    if 'Measure Name' in df.columns:
        df = pd.get_dummies(df, columns=['Measure Name'], drop_first=True)

    drop_cols = ['Facility ID']
    df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True)

    return df


def get_features_and_target(df):
    target_col = 'Excess Readmission Ratio'
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return X, y