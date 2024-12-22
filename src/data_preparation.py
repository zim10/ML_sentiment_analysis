import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
def load_data(filepath):
    """load dataset from file"""
    return pd.read_csv(filepath)

def clean_data(df):
    """Remove missing values and outliers"""
    return df.dropna()

def scale_features(df):
    """scale numerical features"""
    scaler = StandardScaler()
    return scaler.fit_transform(df)