#MPreprocessing - Multiencode.py
#%%
##Load packages
from src.epo_utils.data_loader import process_and_save_data
from src.epo_utils.preprocessing import *
from src.epo_utils.summary_stats import *
import pandas as pd
import numpy as np
import ast

#%% Load Data

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
#%%
df['patent revoked'] = (df['Order status'] == 'patent revoked').astype(int)
#%% Filter to target columns of interest for cleaning thats needed
cols = ['Decision date', 'IPC pharma', 'IPCs',
           'Language', 'Patent Proprietor', 'Headword',
           'Keywords', 'Decisions cited', 'Opponents','patent revoked']

df=df[cols]
#%% resolve IPC separators, drop subclasses
df['IPCs'] = df['IPCs'].str.replace('/', '-', regex=False).str.replace(',', '/', regex=False)
df['IPCs'] = df['IPCs'].str.replace(r'-[^/]+', '', regex=True)

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
#%% get all remaining non-date and non-int columns for further  transformation
categorical_columns = df.select_dtypes(exclude=['int64', 'float64','datetime64[ns]']).columns
#%%
df[categorical_columns].columns
#%%
df[categorical_columns].info()
#%% format list like rows and remove spaces
# Function to strip leading and trailing spaces from each item in the list
def strip_spaces(ipc_list):
    if isinstance(ipc_list, list):
        return [item.strip() if isinstance(item, str) else item for item in ipc_list]
    return ipc_list

# Iterate over each categorical column
for col in categorical_columns:
    # Split the string by '/' into a list, ensuring the operation is performed only on strings
    df[col] = df[col].apply(lambda x: x.split('/') if isinstance(x, str) else x)
    # Apply the strip_spaces function to remove leading and trailing spaces from each item
    df[col] = df[col].apply(strip_spaces)

#%%
df['IPCs'].head()

#%%
df[categorical_columns].info()

#%% Get top N values for categorical columns
top_values_dict = get_top_N_cat_values(df, categorical_columns, N=10)

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
df_encoded=df_encoded[['Decision date', 'IPC pharma', 'numcitations',
       'IPCs_A61K31', 'IPCs_A61K8', 'IPCs_A61K9', 'IPCs_A61K39',
       'IPCs_A61KNone', 'Language_EN', 'Language_DE', 'Language_FR',
       'Patent Proprietor_N.V. Nutricia',
       'Patent Proprietor_Société des Produits Nestlé S.A.',
       'Patent Proprietor_BASF SE', "Patent Proprietor_L'Oréal",
       'Patent Proprietor_Novartis AG', 'Headword_NOVARTIS',
       'Headword_NUTRICIA', "Headword_L'OREAL",
       'Headword_BOEHRINGER INGELHEIM', 'Keywords_Inventive step - (no)',
       'Keywords_Basis of decision - text or agreement to text withdrawn by patent proprietor, Basis of decision - patent revoked',
       'Keywords_Inventive step - (yes)',
       'Keywords_Basis of decision - text or agreement to text withdrawn by patent proprietor', 'Opponents_', 'Opponents_Henkel AG & Co. KGaA',
       'Opponents_ARKEMA FRANCE', 'Opponents_N.V. Nutricia',
       'Opponents_BASF SE', 'patent revoked']]
# %% export using path configuration
#export_path = 'C:\\Users\\Rachael\\Documents\\Sandoz Assignment\\data\\processed'  
export_path ='C:\\Users\\RBurris\\OneDrive - Muscular Dystrophy Association\\Documents\\githubtest\\patent-revocation-prediction\\data\\processed'

# Define the output file path
export_file = os.path.join(export_path, "encoded_data_10.xlsx")  # or ".csv" if you prefer CSV

# Export the DataFrame to the file
df_encoded.to_excel(export_file, index=False)  # use .to_csv for CSV

print(f"Exported file to: {export_file}")
# %%
len(df_encoded)
# %%
