
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# Initialize the Logistic Regression model
log_reg_model = LogisticRegression(max_iter=1000, random_state=42)

# Load the test data
test_file_path = '/mnt/data/test.csv'
test_data = pd.read_csv(test_file_path)

# Remove the 'id' column from test data and apply scaling
X_test = test_data.drop('id', axis=1)
X_test_scaled = scaler.transform(X_test)

# Predict probabilities on the test set
test_probabilities = log_reg_model.predict_proba(X_test_scaled)[:, 1]

# Prepare the submission file
submission = pd.DataFrame({'id': test_data['id'], 'smoking': test_probabilities})
submission.head()
