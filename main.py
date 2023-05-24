import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Load the data from the CSV file
data = pd.read_csv('./cars_clus.csv')

# 1- Define the values to convert to NaN
values_to_convert = ['null', '$null$', 'NA', 'NaN', 'missing']

# 3- Convert specified values to NaN
data.replace(values_to_convert, np.nan, inplace=True)

for column in data.columns:
    # Check if column is numeric fill it with mean

    if data[column].dtype != object:
        print(data[column].dtype)
        data[column].fillna(data[column].mean(), inplace=True)
    # For non-numeric columns, fill missing values with the mode
    else:
        data[column].fillna(data[column].mode()[0], inplace=True)

# SCALING OF DATA TO NORMALIZE
features_to_convert_inFloat = ['sales', 'resale', 'price', 'engine_s', 'wheelbas','width','length','curb_wgt','fuel_cap','lnsales']
data[features_to_convert_inFloat] = data[features_to_convert_inFloat].astype(float)
features_to_convert_inInt = ['type', 'horsepow', 'mpg', 'partition']
data[features_to_convert_inInt] = data[features_to_convert_inInt].astype(float).astype(int)

# Standardization
features_to_scale=data.copy()
features_to_scale.drop(["model","manufact"],axis=1,inplace=True)
scaler = StandardScaler()
data_scaled = scaler.fit_transform(features_to_scale)

# Create a new DataFrame with the scaled data
data_scaled = pd.DataFrame(data_scaled, columns=features_to_scale.columns)

# FINDING CORRELATION
corr_matrix = data_scaled.corr()

plt.figure(figsize=(16, 6))
heatmap = sns.heatmap(corr_matrix, cmap="YlGnBu", annot=True)
heatmap.set_title('Correlation Heatmap', fontdict={'fontsize':12}, pad=12);
plt.show()

# Select upper triangle of correlation matrix
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

to_drop = [column for column in upper.columns if any(upper[column] > 0.5)]

# Drop features
data_scaled.drop(to_drop, axis=1, inplace=True)
data_scaled.head()

# Determine the optimal number of clusters (example using the elbow method)
inertias = []
max_clusters = 10
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k,n_init=10, random_state=42)
    kmeans.fit(data_scaled)
    inertias.append(kmeans.inertia_)
# Plot the elbow curve

plt.plot(range(1, 11), inertias, marker='o')
plt.xlabel('Number of Clusters')
plt.ylabel('Within-Cluster Sum of Squares (Inertia)')
plt.title('Elbow Method')
plt.show()

# Choose the optimal number of clusters and fit the K-means model
k = 4
kmeans = KMeans(n_clusters=k,n_init=10, random_state=42)
kmeans.fit(data_scaled)


# Assign the cluster labels to the original data
data_scaled['clusters'] = kmeans.labels_
data_scaled['clusters'] = data_scaled['clusters'].astype(int)
cluster_means = data_scaled.groupby('clusters').mean()
print(cluster_means)
print(data_scaled)



# Visualization
sns.scatterplot(x=data['manufact'], y=data_scaled['clusters'], hue=data_scaled['clusters'])
plt.xlabel('Manufact')
plt.ylabel('Clusters')
plt.title('Clustering Analysis')
plt.xticks(fontsize=5)
plt.yticks(fontsize=5)
plt.figure(figsize=(12, 8))
plt.show()


