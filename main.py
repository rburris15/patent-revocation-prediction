#Main.py
#%% Load Data
from src.epo_utils.data_loader import process_and_save_data
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from src.epo_utils.feature_selection import get_significant_features
from src.epo_utils.preprocessing import create_preprocessor
from sklearn.preprocessing import OneHotEncoder

# Data Loader
json_path = 'data/BOA_database_for_exercise_from_2020.json'  # path to your JSON file
df = process_and_save_data(json_path)
print(df.shape)


# %%
