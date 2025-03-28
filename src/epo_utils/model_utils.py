# src/epo_utils/model_utils.py
import joblib
from sklearn.model_selection import GridSearchCV

# Function to train and tune a model
def train_model(model, param_grid, X_train, y_train, cv=5, scoring='accuracy', n_jobs=-1):
    grid_search = GridSearchCV(model, param_grid, cv=cv, scoring=scoring, n_jobs=n_jobs, verbose=1)
    grid_search.fit(X_train, y_train)
    return grid_search.best_estimator_, grid_search.best_params_

# Function to save models
def save_model(model, filename):
    joblib.dump(model, filename)

# Function to make predictions
def make_predictions(model, X_test):
    return model.predict(X_test)