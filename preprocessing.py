#Main.py
#%%
##Load Data
from src.epo_utils.data_loader import process_and_save_data
from src.epo_utils.preprocessing import *
from src.epo_utils.summary_stats import *
import pandas as pd
import numpy as np
import ast

#%% Load Data and Preprocess

# Data Loader
json_path = 'data/BOA_database_for_exercise_from_2020.json'  # path to your JSON file
df = process_and_save_data(json_path)
print(df.shape)

#%% preview a summary
summary=generate_summary_table(df)
summary
#%%
"""From this table, the data needs further refining. 
    1. some case & applications numbers are not unique and need to be reviewed and deduplicated
    2. IPC biosimilar does not contain identifying information.
    3. many columns contains multiple values and various separators
    4. some columns are not unique in their nature (order vs order status , opponents vs opponent 1-20)
    5. dates are not correctly formatted"""
#%% Filter to target columns of interest for cleaning thats needed
targets = ['Decision date', 'IPC pharma', 'IPCs',
           'Language', 'Patent Proprietor', 'Headword',
           'Keywords', 'Decisions cited', 'Order status', 'Opponents']

df=df[targets]
#%% resolve IPC separators
df['IPCs'] = df['IPCs'].str.replace('/', '-', regex=False).str.replace(',', '/', regex=False)

#%%
df['IPCs'].head()
#%% date formatting
df['Decision date'] = pd.to_datetime(df['Decision date'], errors='coerce')
#%% create a count of citations from decisions cited
df['numcitations'] = df['Decisions cited'].apply(lambda x: len(x) if isinstance(x, list) else 0)
df.drop(columns=['Decisions cited'], inplace=True) #remove orginal column
#%% preview new summary
summary=generate_summary_table(df)
summary
#%% get all non-date and non-int columns for further  transformation
categorical_columns = df.select_dtypes(exclude=['int64', 'float64','datetime64[ns]']).columns
categorical_columns
#%%
for col in categorical_columns:
    df[col] = df[col].apply(lambda x: np.array(x.split('/')) if isinstance(x, str) else x)

#%% Remove spaces from the categorical columns
df = strip_spaces(df, categorical_columns)
#%%
df['IPCs'].head()

#%% Get top N values for categorical columns
top_values_dict = get_top_N_cat_values(df, categorical_columns, N=5)

print(top_values_dict)

#%% view stats
stats=categorical_descriptive_stats(df, categorical_columns)
stats

#%%
# Apply Multi-Hot Encoding
df_encoded = multi_hot_encode(df, categorical_columns, top_values_dict)

# %% drop the original columns
df_encoded.drop(columns=categorical_columns, inplace=True)
# %%
df_encoded.columns
# %% export using path configuration

export_path = 'C:\\Users\\Rachael\\Documents\\Sandoz Assignment\\data\\processed'  

# Define the output file path
export_file = os.path.join(export_path, "encoded_data.xlsx")  # or ".csv" if you prefer CSV

# Export the DataFrame to the file
df_encoded.to_excel(export_file, index=False)  # use .to_csv for CSV

print(f"Exported file to: {export_file}")
# %%
