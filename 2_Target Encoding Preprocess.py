#Preprocessing -  Target encode.py
# %%
## Load packages
from src.epo_utils.data_loader import process_and_save_data
from src.epo_utils.preprocessing import *
from src.epo_utils.summary_stats import *
import pandas as pd
import numpy as np
import ast
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

# %% Ingest Data

# Prepare paths for ingesting process data
json_path = 'data/BOA_database_for_exercise_from_2020.json'
df = process_and_save_data(json_path)
print(df.shape)

# %% preview a summary
summary = generate_summary_table(df)
summary

#%% export summary
export_path= 'output\\' #save for use later on additional outputs

export_file = os.path.join(export_path, 'dataset_description_full.xlsx') 

# Export the DataFrame to the file
summary.to_excel(export_file, index=True)

# %%
"""From this table and EDA findings, the data needs further refining before modelling.
    1. some case & applications numbers are not unique and need to be reviewed and deduplicated
    2. IPC biosimilar does not contain identifying information.
    3. many columns contain multiple values and various separators
    4. some columns are not unique in their nature (order vs order status , opponents vs opponent 1-20)
    5. dates are not correctly formatted
    6. we know many categorical columns do not correlate with the outcome and may be dropped"""
# %% create target column
df['patent revoked'] = (df['Order status'] == 'patent revoked').astype(int)

# %% Filter to target columns of interest for cleaning
cols = ['IPC pharma', 'Keywords', 'Decisions cited', 'Opponents', 'patent revoked'] #'Decision date','IPCs', 'Language', 'Patent Proprietor', 'Headword',
df = df[cols]

# %% create a count of citations from decisions cited
# Function to count the number of citations based on delimiters (comma or space)
def count_citations(citation_str):
    # Split by common delimiters (e.g., comma, space, slash)
    citations = citation_str.split(",")  # Split by comma first
    citations = [citation.strip() for citation in citations if citation.strip()]  # Strip any extra spaces and remove empty strings
    return len(citations)

# Apply the count_citations function to the 'Decisions cited' column
df['numcitations'] = df['Decisions cited'].apply(lambda x: count_citations(x) if isinstance(x, str) else 0)

# Check the results
df[['Decisions cited', 'numcitations']].head()
#df.drop(columns=['Decisions cited'], inplace=True)  # remove original column
# %% preview new summary
summary = generate_summary_table(df)
#%% export summary
export_path= 'output\\' #save for use later on additional outputs

export_file = os.path.join(export_path, 'dataset_description_targets.xlsx') 

# Export the DataFrame to the file
summary.to_excel(export_file, index=False)

# %% Get all remaining non-date and non-int columns for further transformation
categorical_columns = df.select_dtypes(exclude=['int64', 'float64']).columns

# %% Format list-like rows and remove spaces
# Function to strip leading and trailing spaces from each item in the list
def strip_spaces(ipc_list):
    if isinstance(ipc_list, list):
        return [item.strip() if isinstance(item, str) else item for item in ipc_list]
    return ipc_list

# Iterate over each categorical column and flatten list-like columns
for col in categorical_columns:
    if df[col].apply(lambda x: isinstance(x, list)).any():  # Check if the column contains lists
        # Flatten the lists into a single category string, separate by a unique delimiter (e.g., ' ')
        df[col] = df[col].apply(lambda x: ' '.join(x) if isinstance(x, list) else x)
    # Apply the strip_spaces function to remove leading and trailing spaces from each item
    df[col] = df[col].apply(strip_spaces)

# %% Target encoding for categorical columns
target = 'patent revoked'  # The target variable

# Target encoding function with dropping of original categorical columns
def target_encode(df, categorical_cols, target):
    # Apply target encoding to each categorical column
    for col in categorical_cols:
        if df[col].dtype == 'object':  # Ensure it's categorical (string type)
            encoding = df.groupby(col)[target].mean()  # Compute the mean of target for each category
            df[f'{col}_encoded'] = df[col].map(encoding)  # Map the mean to each row based on the category
    
    # Drop original categorical columns except for the target variable
    columns_to_drop = [col for col in categorical_cols if col != target]
    df.drop(columns=columns_to_drop, inplace=True)

    return df

# Apply target encoding to the categorical columns and drop the categorical columns
df_encoded = target_encode(df.copy(), categorical_columns, target)

# Check if the target is included and the categorical columns are dropped
df_encoded.head()
#%%
df_encoded.columns

#%%
df_encoded.info()
#%% check correlation of new dataset
# Select numeric columns 
numeric_columns = df_encoded.select_dtypes(include=['int64', 'float64'])
#%% Plot the heatmap
import seaborn as sns

# Compute the correlation matrix
correlation_matrix = numeric_columns.corr()

# Generate a mask for the upper triangle
mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(12, 10))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(230, 20, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(correlation_matrix, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})

plt.title('Correlation Matrix for Numeric and Encoded Columns')
plt.show()
#%%
correlation_matrix

#%% export correlation matrix
export_path= 'output\\' #save for use later on additional outputs

export_file = os.path.join(export_path, 'correlation_matrix.xlsx') 

# Export the DataFrame to the file
correlation_matrix.to_excel(export_file, index=True)
#%% check keywords for overlap with target information - may be reported AFTER case outcome is determined and contain revocation status
"""Keywords seem to have a high correlation with both opponents and decisions so it is likely redundant information. we will drop for modelling"""
df['Keywords'].value_counts()
#%%
df_encoded = df_encoded.drop(['Keywords_encoded'], axis=1)
#%%
df_encoded.info()

#%% check dataset balance
print(df_encoded['patent revoked'].value_counts(normalize=True))
sns.countplot(x=df_encoded['patent revoked'])
plt.title("Class Distribution")
plt.show()
"""Dataset balance is within an acceptable ratio. no balancing techniques needed"""
#%% get final summary info on processed data
summary = generate_summary_table(df_encoded)
#%% export summary
export_path= 'output\\' #save for use later on additional outputs

export_file = os.path.join(export_path, 'dataset_description_encoded_data.xlsx') 

# Export the DataFrame to the file
summary.to_excel(export_file, index=True)

# %% Export processed data using path configuration
export_path = 'data\\processed' #repo location for processed data

# Define the output file path
export_file = os.path.join(export_path, "encoded_data_target_encoding.xlsx")  

# Export the DataFrame to the file
df_encoded.to_excel(export_file, index=True)  

print(f"Exported file to: {export_file}")
# %%
df_encoded['patent revoked'].value_counts()
# %%
