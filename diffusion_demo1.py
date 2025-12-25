# %%
import sys
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
xc = mesh.barycenters[:,0]
yc = mesh.barycenters[:,1]
zc = mesh.barycenters[:,2]
#u = (xc**2 + yc**2 + zc**2 + (xc*yc*zc)**2)
gamma = 0.1+0.5*(1+np.tanh(5*(1-(xc*yc*zc)**2*27*2)))
phi = 1.0/np.cosh(100*(xc - 1.0))

print(f"Edge lengths: min={min(mesh.edge_lengths):.6f}, max={max(mesh.edge_lengths):.6f}")

J_diff,rhs_diff = mesh.diffusion_matrix(phi, gamma)
Iv = mesh.identity_matrix()

# %%
# Progress bar utility
def progress_bar(iterable, total=None, desc="Processing"):
    if total is None:
        try:
            total = len(iterable)
        except TypeError:
            total = 0

    # Check if we are in a TTY
    is_tty = sys.stdout.isatty()

    for i, item in enumerate(iterable):
        yield item
        if is_tty and total > 0:
            percent = (i + 1) / total
            bar_length = 40
            filled_length = int(bar_length * percent)
            bar = '█' * filled_length + '-' * (bar_length - filled_length)
            sys.stdout.write(f'\r{desc}: |{bar}| {percent:.1%}')
            sys.stdout.flush()

    if is_tty:
        sys.stdout.write('\n')

with meshio.xdmf.TimeSeriesWriter("results/diffusion_test.xdmf") as writer:
    writer.write_points_cells(points, [("triangle", simplices),])
    for t in progress_bar(range(101), desc="Simulating Diffusion"):
        # Solve
        dt = 0.01
        phi = spla.gmres(Iv+dt*J_diff, phi*mesh.areas+dt*rhs_diff)[0]
        writer.write_data(t*dt, cell_data={"phi": [phi]})
