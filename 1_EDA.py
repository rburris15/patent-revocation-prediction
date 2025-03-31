#EDA.py

##Load Packages
#%%
from src.epo_utils.data_loader import process_and_save_data
import pandas as pd
import numpy as np
#%% Load Data
json_path = 'data/BOA_database_for_exercise_from_2020.json'  # path to data
df = process_and_save_data(json_path)
print(df.shape)

## EDA
#%%
# Data types of each column
print(df.dtypes)

# Summary statistics (only for numeric columns)
print(df.describe())
#%% Time Frame for Dataset
earliest_date = df['Decision date'].min()
latest_date = df['Decision date'].max()  # Get the earliest date
print(f"The earliest date is: {earliest_date}")
print(f"The earliest date is: {latest_date}")
#%% row count
len(df)
#%% Missing Values
# Count missing values per column
print(df.isnull().sum().sort_values(ascending=False))

# Percentage of missing values
missing_percentage = df.isnull().sum() / len(df) * 100
print(missing_percentage[missing_percentage > 0])
#%% Decisions Per Year
df["Decision date"] = pd.to_datetime(df["Decision date"], errors="coerce")

# Extract year and analyze decisions over time
df["Year"] = df["Decision date"].dt.year
df["Year"].value_counts().sort_index().plot(kind="bar", title="Number of Decisions Per Year")
#%% Most Common IPCs
print(df["IPC pharma"].value_counts().head(10))
print(df["IPC biosimilar"].value_counts().head(10))
print(df["IPCs"].explode().value_counts().head(10))  
#%% Language Distribution
df["Language"].value_counts().plot(kind="bar", title="Languages Used in Cases")

#%% Top Originators
topprop=df["Patent Proprietor"].value_counts().head(10)
topprop.head(10).plot(kind="bar", title="Top 10 Patent Proprietors")
#%% export summary
export_path= 'output\\' #save for use later on additional outputs

export_file = os.path.join(export_path, 'Patent Proprietors.xlsx') 

# Export the DataFrame to the file
topprop.to_excel(export_file, index=True)
#%% Keywords
from collections import Counter

# Flatten list if stored as strings with commas
keywords = df["Keywords"].dropna().str.split(", ").explode()
provisions = df["Provisions"].dropna().str.split(", ").explode()

# Count occurrences
print(Counter(keywords).most_common(10))
print(Counter(provisions).most_common(10))

#%% Order Status
outcomes=df["Order status"].value_counts()
outcomes.plot(kind="bar", title="Case Order Status Distribution")
export_file = os.path.join(export_path, 'Case Outcomes.xlsx') 
# Export the DataFrame to the file
topprop.to_excel(export_file, index=True)
#%% Decision Types
from wordcloud import WordCloud
import matplotlib.pyplot as plt

text = " ".join(df["Summary"].dropna())

# Generate word cloud
wordcloud = WordCloud(width=800, height=400, background_color="white").generate(text)

plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()

## statistical EDA
#%% 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency
# %% create Target column
df['patent revoked'] = (df['Order status'] == 'patent revoked').astype(int)
#%% Function to perform Chi-Square test and return p-values
def chi_square_test(df, categorical_columns, target_column):
    p_values = {}
    
    for col in categorical_columns:
        # Explode list-like columns into separate rows
        df_exploded = df.explode(col)
        
        # Create a contingency table for the feature and target variable
        crosstab = pd.crosstab(df_exploded[col], df_exploded[target_column])
        
        # Perform the Chi-Square test
        chi2, p, dof, expected = chi2_contingency(crosstab)
        
        # Store the p-value in the dictionary
        p_values[col] = p
    
    return p_values

#%% Target variable is 'patent revoked'
categorical_columns = ['IPC pharma', 'IPCs', 'Language', 'Patent Proprietor', 'Headword', 'Keywords', 'Opponents']  # Modify based on your dataset
target_column = 'patent revoked'

#%% Run Chi-Square test
p_values = chi_square_test(df, categorical_columns, target_column)

#%% Convert the p-values to a DataFrame for easy manipulation
p_values_df = pd.DataFrame(list(p_values.items()), columns=['Feature', 'p_value'])

#%% Filter significant features (p-value < 0.05)
significant_p_values = p_values_df[p_values_df['p_value'] < 0.05]

#%% Print the significant features and their p-values
print("Significant Features (p-value < 0.05):")
print(significant_p_values)

#%% Plot the p-values for significant features
plt.figure(figsize=(10, 6))
sns.barplot(x='p_value', y='Feature', data=significant_p_values, palette='viridis')
plt.title("Significant Features with Chi-Square Test p-values")
plt.xlabel("p-value")
plt.ylabel("Feature")
plt.show()
#%% numeric correlation
df.columns
#%%
# Select only numeric columns
numeric_columns = df.select_dtypes(include=['int64', 'float64'])

# Compute the correlation matrix
correlation_matrix = numeric_columns.corr()

# Display the correlation matrix
print(correlation_matrix)

# Plot the heatmap
plt.figure(figsize=(10, 8))
# %% view top categorical values for EDA
from src.epo_utils.preprocessing import get_top_N_cat_values
top=get_top_N_cat_values(df,categorical_columns=categorical_columns, N=5)

top
# %%
