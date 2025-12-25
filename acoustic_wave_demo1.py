# %%
import sys
import numpy as np
from scipy.spatial import Delaunay, minkowski_distance
import meshio
from icosphere import icosphere
from mesh import Mesh

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
u = 1.0/np.cosh(100*(xc - 1.0))
v = 0*xc

print(f"Edge lengths: min={min(mesh.edge_lengths):.6f}, max={max(mesh.edge_lengths):.6f}")

mesh0 = meshio.Mesh(
    points,
    [("triangle", simplices),],
    cell_data={"u": [u],"v": [v],"c": [np.sqrt(gamma)]},
)
mesh0.write(
    "wave_test0.vtk",  # str, os.PathLike, or buffer/open file
    # file_format="vtk",  # optional if first argument is a path; inferred from extension
)

J_diff,rhs_diff = mesh.diffusion_matrix(u, gamma)

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

with meshio.xdmf.TimeSeriesWriter("results/wave_test.xdmf") as writer:
    writer.write_points_cells(points, [("triangle", simplices),])
    for t in progress_bar(range(501), desc="Simulating Wave"):
        # Solve
        dt = 0.01
        v = v - dt*J_diff@u/mesh.areas
        u = u + dt*v
        if t % 10 == 0:
            writer.write_data(t, cell_data={"u": [u],"v": [v],"c": [np.sqrt(gamma)]})
