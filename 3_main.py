#Main.py
#%% Load Data

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV
import joblib
import json
#%%
# Load Data
file_path = 'data\processed\encoded_data_target_encoding.xlsx'  # path to processed and encoded data from previous step
df= pd.read_excel(file_path,index_col=0)
#%% preview columns and confirm data is as expected
df.info()

#%% Define target and features
target_column = "patent revoked"
features = [col for col in df.columns if col != target_column]

#%% Split dataset for machine learning
X = df[features]
y = df[target_column]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

## Machine Learning with GridSearch
#%% Define parameter grids for tuning
param_grid_rf = {
    'n_estimators': [50, 100, 200],
    'max_depth': [10, 20,30],
    'min_samples_split': [2, 5, 10],
}

param_grid_xgb = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 6, 9],
    'learning_rate': [ 0.1, 0.2,0.3],
    'subsample': [0.8, 1.0, 1.2],
}

#%% Initialize classic models
rf_model = RandomForestClassifier(random_state=42)
xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)

#%% Grid Search for Random Forest
grid_rf = GridSearchCV(rf_model, param_grid_rf, cv=5, scoring='accuracy', n_jobs=-1, verbose=1)
grid_rf.fit(X_train, y_train)

#%% Grid Search for XGBoost
grid_xgb = GridSearchCV(xgb_model, param_grid_xgb, cv=5, scoring='accuracy', n_jobs=-1, verbose=1)
grid_xgb.fit(X_train, y_train)

#%% best performing models
best_rf = grid_rf.best_estimator_
best_xgb = grid_xgb.best_estimator_
#%%
best_xgb
#%% Save trained models
joblib.dump(best_rf, "models/best_random_forest_TargetEncoded.pkl")
joblib.dump(best_xgb, "models/best_xgboost_TargetEncoded.pkl")
#%% Make predictions
y_pred_rf = best_rf.predict(X_test)
y_pred_xgb = best_xgb.predict(X_test)

#%% Evaluate models
rf_metrics = {
    "accuracy": accuracy_score(y_test, y_pred_rf),
    "classification_report": classification_report(y_test, y_pred_rf, output_dict=True),
    "best_params": grid_rf.best_params_,
}

xgb_metrics = {
    "accuracy": accuracy_score(y_test, y_pred_xgb),
    "classification_report": classification_report(y_test, y_pred_xgb, output_dict=True),
    "best_params": grid_xgb.best_params_,
}

# %% save metrics
from src.epo_utils.metric_utils import *
import os

# Define export path
export_path = 'output\\'  
os.makedirs(export_path, exist_ok=True)  # Ensure the directory exists

# Dictionary of models and their corresponding metrics
models_metrics = {
    "random_forest": rf_metrics,
    "xgboost": xgb_metrics
}

# Iterate through each model's metrics, process, and save
for model_name, metrics in models_metrics.items():
    file_path = os.path.join(export_path, f"{model_name}_metrics.xlsx")
    combined_metrics = combine_metrics(metrics)
    save_metrics_to_csv(combined_metrics, file_path)

print("✅ Metrics for both models saved successfully!")

## PyTorch Neural Network
# %% 
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler

#%% Convert dataframe to numpy
X = df.drop(columns=['patent revoked']).values
y = df['patent revoked'].values

#%% Train-validation-test split (80-10-10)
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

#%% Normalize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

#%% Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

#%% Create data loaders
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# %% Define New Neural Network Model
class PatentRevocationNN(nn.Module):
    def __init__(self, input_size):
        super(PatentRevocationNN, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.3),

            nn.Linear(128, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.3),

            nn.Linear(64, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Dropout(0.3),

            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

# %% Initialize the model
input_size = X_train.shape[1]
model = PatentRevocationNN(input_size)

# %% Define loss function and optimizer
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# %% Learning Rate Scheduler
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)

# %% Early Stopping Setup
best_val_loss = float("inf")
patience = 5  # Stop training if validation loss doesn't improve for 5 epochs
counter = 0

# %% Training Loop with Early Stopping
epochs = 100
for epoch in range(epochs):
    model.train()
    total_loss = 0

    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        y_pred = model(X_batch)
        loss = criterion(y_pred, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    # Validation phase
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for X_val_batch, y_val_batch in val_loader:
            y_val_pred = model(X_val_batch)
            val_loss += criterion(y_val_pred, y_val_batch).item()

    val_loss /= len(val_loader)
    print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss:.4f}, Val Loss: {val_loss:.4f}")

    # Adjust learning rate based on validation loss
    scheduler.step(val_loss)

    # Early stopping check
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        counter = 0  # Reset patience
        torch.save(model.state_dict(), "models/best_patent_nn.pth")  # Save best model
    else:
        counter += 1
        if counter >= patience:
            print("Early stopping triggered!")
            break  # Stop training

# %% Load best model before evaluation
model.load_state_dict(torch.load("models/best_patent_nn.pth"))
model.eval()

# %% Evaluate on test data
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
#%%
y_test_preds = []
with torch.no_grad():
    for X_test_batch, _ in test_loader:
        y_test_preds.append(model(X_test_batch))

y_test_preds = torch.cat(y_test_preds, dim=0).numpy() > 0.5
test_accuracy = np.mean(y_test_preds.flatten() == y_test)

print(f"Test Accuracy: {test_accuracy:.4f}")
print(classification_report(y_test, y_test_preds))

# Create a dictionary to store test metrics
nn_metrics = {
    'test_accuracy': test_accuracy,
    'classification_report': classification_report(y_test, y_test_preds, output_dict=True)  # Saving as dictionary
}

# Save the metrics
with open('neural_network_metrics.json', 'w') as f:
    json.dump(test_metrics, f, indent=4)
#%% save model
joblib.dump(nn, "models/nn_earlystopping_TargetEncoded.pkl")

