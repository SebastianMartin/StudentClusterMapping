from scipy.spatial import ConvexHull
import numpy as np
pointa = np.random.rand(30, 2)   # 30 random points in 2-D
points = []
for point in pointa:
	print(type(point))
	points.append(point)
for point in points:
	print(type(points))


print(points)
print(type(points))
hull = ConvexHull(points)
import matplotlib.pyplot as plt
plt.plot(points[:,0], points[:,1], 'o')
for simplex in hull.simplices:  
	plt.plot(points[simplex, 0], points[simplex, 1], 'k-')
plt.plot(points[hull.vertices,0], points[hull.vertices,1], 'r--', lw=2)
plt.plot(points[hull.vertices[0],0], points[hull.vertices[0],1], 'ro')
print(hull.area)
plt.show()