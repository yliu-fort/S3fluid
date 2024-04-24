# %%
import numpy as np
from scipy.spatial import Delaunay, minkowski_distance
import meshio
from icosphere import icosphere
from mesh import Mesh
import scipy.sparse.linalg as spla

points, simplices = icosphere(30)
point_normals = points / np.linalg.norm(points,axis=-1,keepdims=True)
mesh = Mesh(points, simplices, point_normals)

# %% Init
# \phi = x^2 + y^2 + (xy)^2
xc = mesh.barycenters[0]
yc = mesh.barycenters[1]
zc = mesh.barycenters[2]
#u = (xc**2 + yc**2 + zc**2 + (xc*yc*zc)**2)
gamma = 0.1+0.5*(1+np.tanh(5*(1-(xc*yc*zc)**2*27*2)))
phi = 1.0/np.cosh(100*(xc - 1.0))

print(min(mesh.edge_lengths),max(mesh.edge_lengths))

J_diff,rhs_diff = mesh.diffusion_matrix(phi, gamma)
Iv = mesh.identity_matrix()

# %%
with meshio.xdmf.TimeSeriesWriter("diffusion_test.xdmf") as writer:
    writer.write_points_cells(points, [("triangle", simplices),])
    for t in range(101):
        # Solve
        dt = 0.01
        phi = spla.gmres(Iv+dt*J_diff, phi*mesh.areas+dt*rhs_diff)[0]
        print(t*dt)
        writer.write_data(t*dt, cell_data={"phi": [phi]})
