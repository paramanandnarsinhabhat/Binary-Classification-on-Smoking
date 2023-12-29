import pandas as pd


#Load the training data
train_file_path = '/Users/paramanandbhat/Downloads/playground-series-s3e24/train.csv'
train_data = pd.read_csv(train_file_path)

# Display the first few rows of the training data and summary information
print(train_data.head())
print(train_data.info())
print(train_data.describe())