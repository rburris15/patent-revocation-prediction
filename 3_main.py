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
df= pd.read_excel(file_path)
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

#%% Define Neural Network Model
class PatentRevocationNN(nn.Module):
    def __init__(self, input_size):
        super(PatentRevocationNN, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.3),  # Dropout to prevent overfitting

            nn.Linear(128, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.3),

            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.model(x)

#%% Initialize model
input_size = X_train.shape[1]
nn = PatentRevocationNN(input_size)

#%% Define loss and optimizer
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

#%% Learning Rate Scheduler
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)

#%% Early Stopping Parameters
best_val_loss = float("inf")
patience = 5  # Stop training if validation loss doesn't improve for 5 epochs
counter = 0

#%% Training Loop with Early Stopping
epochs = 100
for epoch in range(epochs):
    nn.train()
    total_loss = 0
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        y_pred = nn(X_batch)
        loss = criterion(y_pred, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    # Validation Phase
    nn.eval()
    val_loss = 0
    with torch.no_grad():
        for X_val_batch, y_val_batch in val_loader:
            y_val_pred = nn(X_val_batch)
            val_loss += criterion(y_val_pred, y_val_batch).item()
    
    val_loss /= len(val_loader)
    print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss:.4f}, Val Loss: {val_loss:.4f}")

    # Reduce LR if validation loss stagnates
    scheduler.step(val_loss)

    # Early Stopping Check
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        counter = 0  # Reset patience
    else:
        counter += 1
        if counter >= patience:
            print("Early stopping triggered!")
            break  # Stop training

#%% Evaluate Model
nn.eval()
with torch.no_grad():
    y_pred_test = nn(X_test_tensor)
    y_pred_test = (y_pred_test.numpy() > 0.5).astype(int)

from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred_test))

#%% save model
joblib.dump(nn, "models/nn_earlystopping_TargetEncoded.pkl")

## Optuna
# #%% with optuna
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
import optuna
import numpy as np

#%% Define the model with tunable hyperparameters
class PatentRevocationNN(nn.Module):
    def __init__(self, input_size, hidden_size, dropout_rate):
        super(PatentRevocationNN, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size),
            nn.Dropout(dropout_rate),

            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size // 2),
            nn.Dropout(dropout_rate),

            nn.Linear(hidden_size // 2, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.model(x)

#%% Define the Optuna objective function
def objective(trial):
    hidden_size = trial.suggest_int("hidden_size", 64, 256, step=32)
    dropout_rate = trial.suggest_float("dropout_rate", 0.1, 0.5, step=0.1)
    learning_rate = trial.suggest_loguniform("learning_rate", 1e-4, 1e-2)
    
    model = PatentRevocationNN(input_size=X_train.shape[1], hidden_size=hidden_size, dropout_rate=dropout_rate)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.BCELoss()

    # Training loop
    for epoch in range(20):  # Run for a smaller number of epochs
        model.train()
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()

    # Validation phase
    model.eval()
    y_val_preds = []
    with torch.no_grad():
        for X_val_batch, _ in val_loader:
            y_val_preds.append(model(X_val_batch))
    
    # Convert predictions to binary values
    y_val_preds = torch.cat(y_val_preds, dim=0)  # Concatenate all predictions
    y_val_preds = y_val_preds.detach().cpu().numpy()  # Detach and move to CPU
    y_val_preds = (y_val_preds > 0.5).astype(int)  # Apply threshold

    y_val = y_val_tensor.detach().cpu().numpy()  # Detach and move validation labels to CPU

    accuracy = accuracy_score(y_val, y_val_preds)
    
    return accuracy  # Optuna maximizes the accuracy

#%% Load and preprocess data 
#X = df.drop(columns=['patent revoked']).values
#y = df['patent revoked'].values

# Train-validation-test split (80-10-10)
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
#val loader
val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
#test loader, for later
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
#%% Run Optuna optimization
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=50)

# Print the best hyperparameters and accuracy
print("Best Hyperparameters:", study.best_params)
print("Best Accuracy:", study.best_value)

# %% Retrieve the best hyperparameters
best_trial = study.best_trial
best_hidden_size = best_trial.params['hidden_size']
best_dropout_rate = best_trial.params['dropout_rate']
best_learning_rate = best_trial.params['learning_rate']

#%% Rebuild and train the best model with these hyperparameters
best_model = PatentRevocationNN(
    input_size=X_train.shape[1],
    hidden_size=best_hidden_size,
    dropout_rate=best_dropout_rate
)

optimizer = optim.Adam(best_model.parameters(), lr=best_learning_rate)
criterion = nn.BCELoss()

# Training loop (same as before with early stopping)
# ...

# Save the best model
torch.save(best_model.state_dict(), "models/best_optuna_nn_model.pth")

#%% Load the saved model for evaluation
# Load the model state_dict into a new instance of the model
loaded_model = PatentRevocationNN(input_size=X_train.shape[1], hidden_size=best_hidden_size, dropout_rate=best_dropout_rate)
loaded_model.load_state_dict(torch.load("models/best_optuna_nn_model.pth"))

#%% Evaluate on the test set using the loaded model
loaded_model.eval()
y_test_preds = []
with torch.no_grad():
    for X_test_batch, _ in test_loader:
        y_test_preds.append(loaded_model(X_test_batch))

y_test_preds = torch.cat(y_test_preds, dim=0).cpu().numpy() > 0.5
test_accuracy = accuracy_score(y_test, y_test_preds)
print(f"Test Accuracy: {test_accuracy:.4f}")

#%% Print classification report for more details
print(classification_report(y_test, y_test_preds))

# %%
