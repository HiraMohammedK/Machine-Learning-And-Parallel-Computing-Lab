import numpy as np
from pyclustering.cluster.kmedoids import kmedoids
import matplotlib.pyplot as plt

X = np.random.rand(100, 2) 
sample = X.tolist()

n_clusters = 3

initial_medoids = np.random.choice(range(len(X)), n_clusters, replace=False)
kmedoids_instance = kmedoids(sample, initial_medoids)

kmedoids_instance.process()

clusters = kmedoids_instance.get_clusters()
medoids = kmedoids_instance.get_medoids()

for i, cluster in enumerate(clusters):
    plt.scatter(X[cluster, 0], X[cluster, 1], label=f'Cluster {i}')
plt.scatter(X[medoids, 0], X[medoids, 1], s=80, color='k', label='Medoids')
plt.legend()
plt.show()
