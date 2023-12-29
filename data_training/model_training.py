from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

# Initialize the Logistic Regression model
log_reg_model = LogisticRegression(max_iter=1000, random_state=42)

# Train the model
log_reg_model.fit(X_train_scaled, y_train)



