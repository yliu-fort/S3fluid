# %%
import numpy as np
from scipy.spatial import Delaunay, minkowski_distance
import meshio
from icosphere import icosphere
from mesh import Mesh

points, simplices = icosphere(120)
point_normals = points / np.linalg.norm(points,axis=-1,keepdims=True)
mesh = Mesh(points, simplices, point_normals)

# %% Init
# \phi = x^2 + y^2 + (xy)^2
xc = mesh.barycenters[:,0]
yc = mesh.barycenters[:,1]
zc = mesh.barycenters[:,2]
#u = (xc**2 + yc**2 + zc**2 + (xc*yc*zc)**2)
gamma = 0.1+0.5*(1+np.tanh(5*(1-(xc*yc*zc)**2*27*2)))
u = 1.0/np.cosh(100*(xc - 1.0))
v = 0*xc

print(min(mesh.edge_lengths),max(mesh.edge_lengths))

mesh0 = meshio.Mesh(
    points,
    [("triangle", simplices),],
    cell_data={"u": [u],"v": [v],"c": [np.sqrt(gamma)]},
)
mesh0.write(
    "wave_test0.vtk",  # str, os.PathLike, or buffer/open file
    # file_format="vtk",  # optional if first argument is a path; inferred from extension
)

# %%
with meshio.xdmf.TimeSeriesWriter("wave_test.xdmf") as writer:
    writer.write_points_cells(points, [("triangle", simplices),])
    for t in range(1000+1):
        # Solve
        dt = 0.01
        J_diff, rhs_diff = mesh.diffusion_matrix(u, gamma)
        v = v - dt*(J_diff@u - rhs_diff)/mesh.areas
        u = u + dt*v
        print(t*dt)
        if t % 10 == 0:
            writer.write_data(t*dt, cell_data={"u": [u],"v": [v],"c": [np.sqrt(gamma)]})
