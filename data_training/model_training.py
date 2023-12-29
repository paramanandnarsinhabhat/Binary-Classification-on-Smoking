from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import sys
import os
import pandas as pd

# Modify the system path to include the directory above the current script.
# This makes it possible to import modules from the parent directory.
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

# Print out the updated system path for debugging purposes.
print("Current sys.path:", sys.path)

# Attempt to import the necessary functions from the Utilities module.
# If the module is not found, print an error message and exit the script.
try:
    from Utilities.common_utilities import split_data, scale_features
except ModuleNotFoundError as e:
    print("Failed to import modules:", e)
    sys.exit(1)


# Define the path to the training data CSV file.
train_file_path = '/Users/paramanandbhat/Downloads/playground-series-s3e24/train.csv'

# Read the training data into a pandas DataFrame.
train_data = pd.read_csv(train_file_path)

# Use the imported split_data function to split the data into training and validation sets.
X_train, X_val, y_train, y_val = split_data(train_data, ['id', 'smoking'], 'smoking')

# Use the imported scale_features function to perform feature scaling on the training and validation sets.
X_train_scaled, X_val_scaled = scale_features(X_train, X_val)

# Initialize the Logistic Regression model
log_reg_model = LogisticRegression(max_iter=1000, random_state=42)

# Train the model
log_reg_model.fit(X_train_scaled, y_train)

# Predict probabilities on the validation set
y_val_prob = log_reg_model.predict_proba(X_val_scaled)[:, 1]

# Calculate ROC AUC score
roc_auc = roc_auc_score(y_val, y_val_prob)
roc_auc
print(roc_auc)


