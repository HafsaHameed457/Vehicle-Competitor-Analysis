import pandas as pd
import numpy as np

# Load the data from the CSV file
data = pd.read_csv('C:/Users/hunai/Desktop/cars_clus.csv')

# DATA_CLEANING

# 1- Drop irrelevant columns
columns_to_drop = ['type', 'lnsales', 'partition']
data.drop(columns_to_drop, axis=1, inplace=True)

# 2- Define the values to convert to NaN
values_to_convert = ['null', '$null$', 'NA', 'NaN', 'missing']

# 3- Convert specified values to NaN
data.replace(values_to_convert, np.nan, inplace=True)

for column in data.columns:
    # Check if column is numeric fill it with mean

    if data[column].dtype != object:
        data[column].fillna(data[column].mean(), inplace=True)
    # For non-numeric columns, fill missing values with the mode
    else:
        data[column].fillna(data[column].mode()[0], inplace=True)

# Print the preprocessed cleaned data
print("Preprocessed Data:")
print(data)
