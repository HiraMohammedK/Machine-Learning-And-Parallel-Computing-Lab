import numpy as np
import matplotlib.pyplot as plt
import skfuzzy as fuzz

X = np.random.rand(100, 2)

n_clusters = 3
cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(X.T, n_clusters, 2, error=0.005, maxiter=1000)

labels = np.argmax(u, axis=0)

for i in range(n_clusters):
    plt.scatter(X[labels == i, 0], X[labels == i, 1], label=f'Cluster {i}')
plt.legend()
plt.show()
