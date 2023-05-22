import pandas as pd
import numpy as np
# Load the data from the CSV file
data = pd.read_csv('C:/Users/hunai/Desktop/cars_clus.csv')

# Drop irrelevant columns
columns_to_drop = ['type', 'lnsales', 'partition']
data.drop(columns_to_drop, axis=1, inplace=True)

# Load the data from the CSV file
data = pd.read_csv('C:/Users/hunai/Desktop/cars_clus.csv')

# Define the values to convert to NaN
values_to_convert = ['null', 'NA', 'NaN', 'missing']

# Convert specified values to NaN
data.replace(values_to_convert, np.nan, inplace=True)


# Print the updated data
# print(data.head())


for column in data.columns:
    if data[column].dtype != object:  # Check if column is numeric
        data[column].fillna(data[column].mean(), inplace=True)
    else:  # For non-numeric columns, fill missing values with the mode
        data[column].fillna(data[column].mode()[0], inplace=True)

# Print the preprocessed data
print("Preprocessed Data:")
print(data)
