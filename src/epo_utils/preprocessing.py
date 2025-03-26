# Function to strip leading and trailing spaces from multiple columns
import pandas as pd
from collections import Counter
from sklearn.preprocessing import MultiLabelBinarizer

def strip_spaces(df, columns):
    """
    Strips leading and trailing spaces from list-like values in the given columns.
    
    Parameters:
    - df (pd.DataFrame): The input DataFrame.
    - columns (list): List of column names to process.

    Returns:
    - pd.DataFrame: DataFrame with spaces removed in list-like values.
    """
    def clean_list(values):
        if isinstance(values, list):  # If it's a list, strip spaces from each element
            return [v.strip() for v in values if isinstance(v, str)]
        return values  # If it's not a list, return as is

    df = df.copy()  # Avoid modifying the original DataFrame
    for col in columns:
        if col in df.columns:
            df[col] = df[col].apply(clean_list)  # Apply cleaning function
    return df


def get_top_N_cat_values(df, categorical_columns, N=10):
    """Generate a dictionary where the key is the column name, 
    and the value is a list of the N most frequent values in that column."""
    
    top_values_dict = {}

    for col in categorical_columns:
        if col in df.columns:
            # Flatten the lists in the column, ignoring NaN values
            flattened_values = df[col].explode().dropna()

            # Count the frequency of each item in the column
            value_counts = Counter(flattened_values)
            
            # Get the top N most common values
            top_n_values = value_counts.most_common(N)
            
            # Store the results in the dictionary (only top N values)
            top_values_dict[col] = [item[0] for item in top_n_values]  # Extracting just the value part of the tuple

    return top_values_dict



def multi_hot_encode(df, categorical_columns, top_values_dict):
    """Encodes the top N values from top_values_dict into binary columns."""
    
    for col in categorical_columns:
        # Ensure that each column contains a list-like value
        if col in df.columns:
            for value in top_values_dict.get(col, []):  # Iterate over the top N values for each column
                # Create a new column for each top value
                df[f'{col}_{value}'] = df[col].apply(lambda x: 1 if value in x else 0)
    
    return df