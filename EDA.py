#EDA.py

##Load Data
#%%
from src.epo_utils.data_loader import process_and_save_data
import pandas as pd
import numpy as np
#%%
# Call the function
json_path = 'data/BOA_database_for_exercise_from_2020.json'  # path to your JSON file
df = process_and_save_data(json_path)
print(df.shape)

## EDA
#%%
# Data types of each column
print(df.dtypes)

# Summary statistics (only for numeric columns)
print(df.describe())
#%% Time Frame for Dataset

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
df["Patent Proprietor"].value_counts().head(10).plot(kind="bar", title="Top 10 Patent Proprietors")

#%% Keywords
from collections import Counter

# Flatten list if stored as strings with commas
keywords = df["Keywords"].dropna().str.split(", ").explode()
provisions = df["Provisions"].dropna().str.split(", ").explode()

# Count occurrences
print(Counter(keywords).most_common(10))
print(Counter(provisions).most_common(10))

#%% Order Status
df["Order status"].value_counts().plot(kind="bar", title="Case Order Status Distribution")

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
