import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

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
        print(data[column].dtype)
        data[column].fillna(data[column].mean(), inplace=True)
    # For non-numeric columns, fill missing values with the mode
    else:
        data[column].fillna(data[column].mode()[0], inplace=True)

# FINDING CORRELATION
features = ['sales','resale','type','price','engine_s','horsepow','wheelbas','width','length','curb_wgt','fuel_cap','mpg',	'lnsales','partition'
            ]
corr_matrix = data[features].corr().abs()

plt.figure(figsize=(16, 6))
heatmap = sns.heatmap(corr_matrix, vmin=-1, vmax=1, annot=True)
heatmap.set_title('Correlation Heatmap', fontdict={'fontsize':12}, pad=12);
# plt.show()

# Select upper triangle of correlation matrix
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))


to_drop = [column for column in upper.columns if any(upper[column] > 0.5)]

# Drop features
data.drop(to_drop, axis=1, inplace=True)

# Print the preprocessed cleaned data
print("Preprocessed Data:")
print(data)
# data.to_csv('dataaa.csv', index=False)

# SCALING OF DATA TO NORMALIZE
# Select the features you want to scale
features_to_scale = ['sales', 'resale', 'type', 'wheelbas', 'partition']
data[features_to_scale] = data[features_to_scale].astype(float)
data_to_scale = data[features_to_scale]

# Standardization
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data_to_scale)

# Create a new DataFrame with the scaled data
# data_scaled = pd.DataFrame(data_scaled, columns=features_to_scale)

# Print the scaled DataFrame

# Determine the optimal number of clusters (example using the elbow method)
inertias = []
silhouette_scores = []
max_clusters = 10
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k,n_init=10, random_state=42)
    kmeans.fit(data_scaled)
    inertias.append(kmeans.inertia_)
    # silhouette_scores.append(silhouette_score(scaled_data, kmeans.labels_))
# Plot the elbow curve

plt.plot(range(1, 11), inertias, marker='o')
plt.xlabel('Number of Clusters')
plt.ylabel('Within-Cluster Sum of Squares (Inertia)')
plt.title('Elbow Method')
# plt.show()

# Choose the optimal number of clusters and fit the K-means model
k = 4
kmeans = KMeans(n_clusters=k,n_init=10, random_state=42)
kmeans.fit(data_scaled)


# Assign the cluster labels to the original data
data['cluster'] = kmeans.labels_
# data['clusters'] = data['clusters'].astype(int)
# data['clusters'] = data['clusters'].str.strip()
# data['cluster'] = pd.to_numeric(int(data['cluster']))
# dataaaa=data['cluster']
#
# data.to_csv('dateee.csv',index=False)
#
cluster_means = data.groupby('cluster').mean()
print(cluster_means)



# Visualization
sns.scatterplot(data=data, x='horsepow', y='price', hue='cluster')
plt.xlabel('Horsepower')
plt.ylabel('Price')
plt.title('Clustering Analysis')
plt.show()


