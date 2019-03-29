from scipy.spatial import ConvexHull
import numpy as np
pointa = np.random.rand(30, 2)   # 30 random points in 2-D
points = []
for point in pointa:
	points.append(point)
pointss = np.asarray(points)

hull = ConvexHull(pointss)
import matplotlib.pyplot as plt
plt.plot(pointss[:,0], pointss[:,1], 'o')
for simplex in hull.simplices:  
	plt.plot(pointss[simplex, 0], pointss[simplex, 1], 'k-')
plt.plot(pointss[hull.vertices,0], pointss[hull.vertices,1], 'r--', lw=2)
plt.plot(pointss[hull.vertices[0],0], pointss[hull.vertices[0],1], 'ro')
print(hull.area)
plt.show()