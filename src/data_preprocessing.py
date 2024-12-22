import pandas as pd
import numpy as np

def preprocess_text(text):
    """Baseic text_preprocessing function"""
    return text.lower().strip()
def load_data(filepath):
    """load and preprocess dataset"""
    df = pd.read_csv(filepath)
    return df