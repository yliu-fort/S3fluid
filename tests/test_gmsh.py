import pygmsh
import meshio
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from icosphere import icosphere


#with pygmsh.geo.Geometry() as geom:
#    geom.add_circle([0.0, 0.0], 1.0, mesh_size=0.2)
#    mesh = geom.generate_mesh()

#with pygmsh.occ.Geometry() as geom:
#    ellipsoid = geom.add_ellipsoid([0.0, 0.0, 0.0], [1.0, 1.0, 1.0],mesh_size=0.2, with_volume=False)
#    mesh = geom.generate_mesh()

#print(mesh.points)
#print(mesh.cells_dict['triangle'])
#print(np.min(mesh.cells_dict['triangle']))
#print(np.max(mesh.cells_dict['triangle']))
#print(mesh)

nu = 7  # or any other integer
points, faces = icosphere(nu)

mesh = meshio.Mesh(
    points,
    [("triangle", faces),],
    # Optionally provide extra data on points, cells, etc.
    #point_data={"T": [0.3, -1.2, 0.5, 0.7, 0.0, -3.0]},
    # Each item in cell data must match the cells array
    #cell_data={"a": [[0.1, 0.2], [0.4]]},
)
print(mesh)

mesh.write(
    "icosphere_test.vtk",  # str, os.PathLike, or buffer/open file
    # file_format="vtk",  # optional if first argument is a path; inferred from extension
)

# Setup for a 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot each simplex
for simplex in faces:
    polygon = points[simplex]
    ax.add_collection3d(Poly3DCollection([polygon], facecolors='grey', linewidths=1, edgecolors='k', alpha=.1))

# Set plot display parameters
ax.scatter(points[:,0], points[:,1], points[:,2], color='k')  # Plot the points
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# Adjusting the scale for better visualization
max_range = np.array([points[:,0].max()-points[:,0].min(), 
                      points[:,1].max()-points[:,1].min(), 
                      points[:,2].max()-points[:,2].min()]).max() / 2.0
mid_x = (points[:,0].max()+points[:,0].min()) * 0.5
mid_y = (points[:,1].max()+points[:,1].min()) * 0.5
mid_z = (points[:,2].max()+points[:,2].min()) * 0.5
ax.set_xlim(mid_x - max_range, mid_x + max_range)
ax.set_ylim(mid_y - max_range, mid_y + max_range)
ax.set_zlim(mid_z - max_range, mid_z + max_range)

plt.show()