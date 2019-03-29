from sklearn.cluster import KMeans
from sklearn import metrics
import numpy as np
from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt

x1 = np.array([3, 1, 1, 2, 1, 6, 6, 6, 5, 6, 7, 8, 9, 8, 9, 9, 8])
x2 = np.array([5, 4, 6, 6, 5, 8, 6, 7, 6, 7, 1, 2, 1, 2, 3, 2, 3])

# create new plot and data
plt.plot()
X = np.array(list(zip(x1, x2))).reshape(len(x1), 2)
colors = ['b', 'g', 'c']
markers = ['o', 'v', 's']

# KMeans algorithm 
K = 3
kmeans_model = KMeans(n_clusters=K).fit(X)

print(kmeans_model.cluster_centers_)
centers = np.array(kmeans_model.cluster_centers_)

plt.plot()
plt.title('k means centroids')

for i, l in enumerate(kmeans_model.labels_):
    plt.plot(x1[i], x2[i], color=colors[l], marker=markers[l],ls='None')
    plt.xlim([0, 10])
    plt.ylim([0, 10])

plt.scatter(centers[:,0], centers[:,1], marker="x", color='r')

labels = kmeans_model.labels_
clusterPoints = [[] for _ in range(K)] 
for i in range(len(labels)):
	clusterPoints[labels[i]].append([x1[i],x2[i]])
for i in range(len(clusterPoints)):
	clusterArray = np.asarray(clusterPoints[i])		
	hull = ConvexHull(clusterArray)
	for simplex in hull.simplices:  
		plt.plot(clusterArray[simplex, 0], clusterArray[simplex, 1],colors[i])
	plt.plot(clusterArray[hull.vertices,0], clusterArray[hull.vertices,1],colors[i] ,label="cluster: ")
plt.gca().legend()

plt.show()