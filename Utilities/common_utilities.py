from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd


def split_data(dataframe, feature_columns_to_drop, target_column, test_size=0.2, random_state=42):
    """
    Splits the data into training and validation sets.

    :param dataframe: The input dataframe.
    :param feature_columns_to_drop: List of column names to be dropped from the features.
    :param target_column: The name of the target column.
    :param test_size: The size of the validation set. Default is 0.2.
    :param random_state: The random state for reproducibility. Default is 42.
    :return: X_train, X_val, y_train, y_val
    """
    X = dataframe.drop(feature_columns_to_drop, axis=1)
    y = dataframe[target_column]

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=test_size, random_state=random_state)
    return X_train, X_val, y_train, y_val


def scale_features(X_train, X_val):
    """
    Applies feature scaling to the training and validation sets.

    :param X_train: Training features.
    :param X_val: Validation features.
    :return: Scaled versions of X_train and X_val.
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    return X_train_scaled, X_val_scaled
