# %% python -m cProfile -o possion.prof profile_possion_solver.py
# %%
import numpy as np
from scipy.spatial import Delaunay, minkowski_distance
import meshio
import pygmsh
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from mesh import Mesh
import scipy.sparse.linalg as spla
from icosphere import icosphere
from geometry import project_vectors_to_planes

np.random.seed(42)

#with pygmsh.geo.Geometry() as geom:
#    geom.add_rectangle(0.0, 1.0, 0.0, 1.0, 0.0, mesh_size=0.01)
#    mesh = geom.generate_mesh()

#points = mesh.points
#simplices = mesh.cells_dict['triangle']
#point_normals = np.array([[0,0,1]]*len(simplices))
#mesh = Mesh(points, simplices, point_normals)

points, simplices = icosphere(30)
point_normals = points / np.linalg.norm(points,axis=-1,keepdims=True)
print(len(simplices))
mesh = Mesh(points, simplices, point_normals)

print(mesh.print_member_shapes())

# %% Init
# \phi = x^2 + y^2 + (xy)^2
xc = mesh.barycenters[0]
yc = mesh.barycenters[1]
zc = mesh.barycenters[2]

phi = 1.0/np.cosh(100*(xc - 0.5))/np.cosh(100*(yc - 0.5))

u = np.random.randn(*(3, len(xc)))
u = project_vectors_to_planes(u,mesh.barynormals)

# Find face flux
for _ in range(10):
    uf = mesh.interpolate_field_cell_to_face(u)
    un = np.sum(uf * mesh.edge_normals, axis = 0)
    un, du = mesh.helmholtz_projection(un)
    u += du
