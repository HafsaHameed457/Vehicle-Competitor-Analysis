import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


# Load the data from the CSV file
data = pd.read_csv('./cars_clus.csv')

# 1- Define the values to convert to NaN
values_to_convert = ['null', '$null$', 'NA', 'NaN', 'missing']

# 3- Convert specified values to NaN
data.replace(values_to_convert, np.nan, inplace=True)
#
for column in data.columns:
    # Check if column is numeric fill it with mean

    if data[column].dtype != object:
        data[column].fillna(data[column].mean(), inplace=True)
    # For non-numeric columns, fill missing values with the mode
    else:
        data[column].fillna(data[column].mode()[0], inplace=True)

# FINDING CORRELATION
features = ['sales','resale','type','price','engine_s','horsepow','wheelbas','width','length','curb_wgt','fuel_cap','mpg',	'lnsales','partition'
            ]
plt.figure(figsize=(16, 6))
heatmap = sns.heatmap(data[features].corr(), vmin=-1, vmax=1, annot=True)
heatmap.set_title('Correlation Heatmap', fontdict={'fontsize':12}, pad=12);
plt.show()

# 1- Drop irrelevant columns
# columns_to_drop = ['sales', 'resale', 'type', 'partition', 'width', 'length', 'curb_wgt', 'fuel_cap', 'mpg']
# data.drop(columns_to_drop, axis=1, inplace=True)

# Print the preprocessed cleaned data
# print("Preprocessed Data:")
# print(data)
# data.to_csv('data.csv', index=False)

# SCALING OF DATA TO NORMALIZE
# Select the features you want to scale
# features_to_scale = ['sales', 'price','engine_s','horsepow','wheelbas']
# data_to_scale = data[features_to_scale]
#
# # Standardization
# scaler = StandardScaler()
# data_scaled = scaler.fit_transform(data_to_scale)
#
# # Create a new DataFrame with the scaled data
# data_scaled = pd.DataFrame(data_scaled, columns=features_to_scale)
#
# # Print the scaled DataFrame
# print(data_scaled)

