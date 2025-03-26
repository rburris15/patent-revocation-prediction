import pandas as pd
def generate_summary_table(df):
    """Generate a summary table for a DataFrame."""
    
    summary = pd.DataFrame({
        'Data Type': df.dtypes,
        'Non-Null Count': df.count(),
        'Null Count': df.isnull().sum(),
        'Null Percentage': (df.isnull().sum() / len(df)) * 100,
        'Unique Values': df.nunique(),
    })
    
    # Get top 5 values for categorical columns (e.g., for insight into distributions)
    top_values = {}
    for col in df.select_dtypes(include=['object']).columns:
        top_values[col] = df[col].value_counts().head(5).to_dict()
    
    summary['Top 5 Values'] = summary.index.map(lambda x: top_values.get(x, {}))
    
    # Get basic stats for numerical columns
    numeric_stats = df.describe().transpose()
    summary = summary.join(numeric_stats[['mean', 'std', 'min', '50%', 'max']], how='left')
    
    return summary

def categorical_descriptive_stats(df, categorical_columns):
    """Generate descriptive statistics for categorical columns."""
    stats = []
    
    for col in categorical_columns:
        if col in df.columns:
            total_rows = len(df)
            
            # Count the number of missing values (NaN) in each row for the specific column
            missing_count = df[col].isna().sum()
            missing_percentage = (missing_count / total_rows) * 100
            missing_values = f"{missing_count} ({missing_percentage:.2f}%)"
            
            # Flatten the lists in each column and drop NaN values
            flattened_values = df[col].explode().dropna()  # Flatten and drop NaN values
            
            # Remove blank values (empty strings)
            flattened_values = flattened_values[flattened_values != '']  # Remove empty strings
            
            unique_values = flattened_values.nunique()
            
            # Most frequent value and its count, ensuring blanks are excluded
            most_frequent = flattened_values.mode()[0] if not flattened_values.mode().empty else None
            most_frequent_count = flattened_values.value_counts().iloc[0] if not flattened_values.value_counts().empty else 0
            
            stats.append({
                "Column": col,
                "Missing Values": missing_values,
                "Unique Values": unique_values,
                "Most Frequent Value": most_frequent,
                "Most Frequent Count": most_frequent_count
            })
    
    return pd.DataFrame(stats)



import pandas as pd

def top_5_unique_values_table(df, columns):
    """Print a table with unique values and top 5 values for each column."""
    result = []
    
    for col in columns:
        if col in df.columns:
            unique_values = df[col].nunique()  # Get the number of unique values
            top_values = df[col].value_counts().head(5)  # Get the top 5 frequent values
            
            # Join top 5 values in a string to display in one cell
            top_values_str = '\n'.join([f"{value}: {count}" for value, count in top_values.items()])
            
            # Append results for each column
            result.append({
                "Column": col,
                "Unique Values": unique_values,
                "Top 5 Values": top_values_str
            })
        else:
            result.append({
                "Column": col,
                "Unique Values": "Column not found",
                "Top 5 Values": "Column not found"
            })
    
    # Convert the result into a DataFrame
    result_df = pd.DataFrame(result)
    
    # Print the table
    return result_df
    
    # Print the table
    print(result_df)
import os
def explode_and_pivot(df, list_columns, application_col, export_path):
    """
    Explodes columns with separated lists, counts unique application numbers per unique value,
    and exports the summary to an Excel file.
    
    Parameters:
    - df: DataFrame containing the data.
    - list_columns: List of column names that contain separated lists.
    - application_col: Column name for application IDs.
    - export_path: Path to save the Excel file.
    
    Returns:
    - None (exports results to Excel)
    """
    
    # Create an Excel writer to store all pivot tables
    output_file = os.path.join(export_path, "pivot_summary.xlsx")
    writer = pd.ExcelWriter(output_file, engine="openpyxl")

    for col in list_columns:
        if col in df.columns:
            # Explode the column while keeping application_col
            df_exploded = df[[application_col, col]].copy()
            df_exploded[col] = df_exploded[col].astype(str).str.split("/")  # Adjust separator if needed
            df_exploded = df_exploded.explode(col)

            # Remove leading/trailing spaces and drop NaN values
            df_exploded[col] = df_exploded[col].str.strip()
            df_exploded = df_exploded.dropna()

            # Create pivot table: Unique values vs count of unique application IDs
            pivot_table = df_exploded.groupby(col)[application_col].nunique().reset_index()
            pivot_table.columns = [col, "Unique Application Count"]

            # Export each pivot table as a separate sheet
            pivot_table.to_excel(writer, sheet_name=col[:30], index=False)

    writer.close()
    print(f"Pivot summary saved to {output_file}")

