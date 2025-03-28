# src/epo_utils/data_utils.py
import pandas as pd
from sklearn.model_selection import train_test_split

# Function to load and preprocess data
def load_data(file_path, target_column):
    df = pd.read_excel(file_path)
    features = [col for col in df.columns if col != target_column]
    X = df[features]
    y = df[target_column]
    return X, y

# Function to split data
def split_data(X, y, test_size=0.2, random_state=42):
    return train_test_split(X, y, test_size=test_size, random_state=random_state)