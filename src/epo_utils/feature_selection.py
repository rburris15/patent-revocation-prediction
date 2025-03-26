# src/feature_selection.py
import pandas as pd
from sklearn.feature_selection import SelectKBest, chi2

from sklearn.feature_selection import chi2
import numpy as np

def chi_squared_test(X_col, y):
    """Performs the Chi-squared test and returns the p-value."""
    X_col = np.array(X_col).reshape(-1, 1)  # Reshape to make it 2D for the chi2 test
    chi2_stat, p_val = chi2(X_col, y)
    return p_val[0]

def get_significant_features(X, y, categorical_columns, alpha=0.05, encoder=None):
    """Identifies the significant features using the Chi-squared test after encoding."""
    if encoder is None:
        raise ValueError("An encoder must be passed to get_significant_features.")
    
    # Encode categorical columns using the provided encoder
    X_encoded = encoder.fit_transform(X[categorical_columns])
    
    # Get the encoded column names
    encoded_column_names = encoder.get_feature_names_out(input_features=categorical_columns)
    
    significant_columns = []
    
    for col in encoded_column_names:
        col_index = list(encoded_column_names).index(col)
        p_value = chi_squared_test(X_encoded[:, col_index], y)
        
        if p_value < alpha:
            significant_columns.append(col)
    
    return significant_columns


