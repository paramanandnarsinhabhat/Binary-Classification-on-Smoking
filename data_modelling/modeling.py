from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd

train_file_path = '/Users/paramanandbhat/Downloads/playground-series-s3e24/train.csv'
train_data = pd.read_csv(train_file_path)
# Separate the features and the target variable
X = train_data.drop(['id', 'smoking'], axis=1)  # Dropping 'id' as it's not a feature
y = train_data['smoking']
