# src/epo_utils/metrics_utils.py
import json
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd
# Function to evaluate model performance
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    class_report = classification_report(y_test, y_pred, output_dict=True)
    return accuracy, class_report

# Function to save metrics to a JSON file
def save_metrics(metrics, filename):
    with open(filename, "w") as f:
        json.dump(metrics, f, indent=4)

def metrics_table(metrics_dict):
    """
    Create a simple table with accuracy and best parameters.
    """
    metrics_data = {
        'Metric': ['Accuracy', 'Best Params (n_estimators)', 'Best Params (max_depth)'],
        'Value': [metrics_dict['accuracy'], metrics_dict['best_params']['n_estimators'], metrics_dict['best_params']['max_depth']],
    }

    metrics_df = pd.DataFrame(metrics_data)
    return metrics_df

# Function to process and flatten the classification report
def process_classification_report(classification_report):
    """
    Convert the classification report into a DataFrame.
    """
    classification_df = pd.DataFrame(classification_report).T  # Transpose to get metrics as rows
    classification_df['Metric'] = classification_df.index
    classification_df = classification_df[['Metric', 'precision', 'recall', 'f1-score', 'support']]
    return classification_df

# Function to combine all metrics into a final DataFrame
def combine_metrics(rf_metrics):
    """
    Combine basic metrics and classification report into a single DataFrame.
    """
    # Create table for basic metrics
    basic_metrics_df = metrics_table(rf_metrics)

    # Process and flatten classification report
    classification_df = process_classification_report(rf_metrics['classification_report'])

    # Combine both tables
    metrics_df = pd.concat([basic_metrics_df, classification_df], ignore_index=True)
    return metrics_df

# Function to save metrics to a CSV file
def save_metrics_to_csv(metrics_df, file_path):
    """
    Save the metrics DataFrame to a CSV file.
    """
    metrics_df.to_csv(file_path, index=False)
    print(f"Metrics saved to {file_path}")