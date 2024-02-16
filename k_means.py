import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

X = np.random.rand(100, 2)  

n_clusters = 3
kmeans = KMeans(n_clusters=n_clusters, init='k-means++', max_iter=300, random_state=42)
kmeans.fit(X)
labels = kmeans.labels_
centroids = kmeans.cluster_centers_
for i in range(n_clusters):
    plt.scatter(X[labels == i, 0], X[labels == i, 1], label=f'Cluster {i}')
plt.scatter(centroids[:, 0], centroids[:, 1], s=80, color='k', label='Centroids')
plt.legend()
plt.show()
