# %%
# %%
import numpy as np
import meshio
import pygmsh
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from mesh import Mesh
import scipy.sparse.linalg as spla
from icosphere import icosphere
from geometry import project_vectors_to_planes

# %%
points, simplices = icosphere(120)
point_normals = points / np.linalg.norm(points,axis=-1,keepdims=True)
print(len(simplices))
mesh = Mesh(points, simplices, point_normals)

if len(simplices) < 5000:
    # Setup for a 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot each simplex
    for simplex in simplices:
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

# %%
# %% Init
# \phi = x^2 + y^2 + (xy)^2
xc = mesh.barycenters[0]
yc = mesh.barycenters[1]
zc = mesh.barycenters[2]

phi = 1.0/np.cosh(100*(xc - 0.5))/np.cosh(100*(yc - 0.5))

xf = mesh.edge_centers[0]
yf = mesh.edge_centers[1]
zf = mesh.edge_centers[2]

u = np.random.randn(*(3,len(xc)))

u = project_vectors_to_planes(u, mesh.barynormals)

Iv = mesh.identity_matrix()

for _ in range(5):
    uf = mesh.interpolate_field_cell_to_face(u)
    un = np.sum(uf * mesh.edge_normals, axis=0)
    un, du = mesh.helmholtz_projection(un)
    u += du

mesh0 = meshio.Mesh(
    points,
    [("triangle", simplices),],
    cell_data={"u": [u.T]},
)
mesh0.write(
    "ns_on_sphere0.vtk",  # str, os.PathLike, or buffer/open file
    # file_format="vtk",  # optional if first argument is a path; inferred from extension
)

# %%
with meshio.xdmf.TimeSeriesWriter("ns_on_sphere.xdmf") as writer:
    writer.write_points_cells(points, [("triangle", simplices),])
    t = 0
    dt = 0.001
    for it in range(501):
        # Projection
        for _ in range(1):
            uf = mesh.interpolate_field_cell_to_face(u)
            un = np.sum(uf * mesh.edge_normals, axis=0)
            un, du = mesh.helmholtz_projection(un)
            u += du

        # Adjust dt according to CFL
        cfl = np.max(np.abs(un)*dt/np.linalg.norm(mesh.barycenters[...,mesh.neighbours] - mesh.barycenters[...,mesh.owners], axis=0))
        
        if cfl < 0.8:
            dt *= 1.2
        elif cfl > 1.5:
            dt /= 2

        # Advection
        crank_nicolson_coeff = 0.5
        uf = un * mesh.edge_normals
        Jx,rhs_x = mesh.convection_matrix(u[0], uf, face_flux=True, quick=True)
        u[0] = spla.bicgstab(Iv+crank_nicolson_coeff*dt*Jx, u[0]*mesh.areas-(1.0-crank_nicolson_coeff)*dt*Jx@u[0]+dt*rhs_x)[0]
        Jy,rhs_y = mesh.convection_matrix(u[1], uf, face_flux=True, quick=True)
        u[1] = spla.bicgstab(Iv+crank_nicolson_coeff*dt*Jy, u[1]*mesh.areas-(1.0-crank_nicolson_coeff)*dt*Jy@u[1]+dt*rhs_y)[0]
        Jz,rhs_z = mesh.convection_matrix(u[2], uf, face_flux=True, quick=True)
        u[2] = spla.bicgstab(Iv+crank_nicolson_coeff*dt*Jz, u[2]*mesh.areas-(1.0-crank_nicolson_coeff)*dt*Jz@u[2]+dt*rhs_z)[0]

        # Project veloity on to the mesh surface
        u = project_vectors_to_planes(u, mesh.barynormals)

        t += dt
        
        if it % 5 == 0:
            print(it, t,"CFL=", cfl)
            writer.write_data(t, cell_data={"u": [u.T]})


