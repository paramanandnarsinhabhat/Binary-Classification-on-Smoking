from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd

train_file_path = '/Users/paramanandbhat/Downloads/playground-series-s3e24/train.csv'
train_data = pd.read_csv(train_file_path)
# Separate the features and the target variable
X = train_data.drop(['id', 'smoking'], axis=1)  # Dropping 'id' as it's not a feature
y = train_data['smoking']

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

# Display the shapes of the training and validation sets
X_train_scaled.shape, X_val_scaled.shape, y_train.shape, y_val.shape
print(X_train_scaled.shape, X_val_scaled.shape, y_train.shape, y_val.shape)

