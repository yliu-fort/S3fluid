import numpy as np
from scipy.sparse import coo_array
import scipy.sparse.linalg as spla
import scipy.linalg as la
from geometry import *

class Mesh(object):
    def __init__(self, points, simplices, point_normals=None) -> None:
        if point_normals.any() == None:
            point_normals = np.array([[0, 0, 1]]*(points.shape[0]))

        self.barycenters = get_barycenters(points, simplices)
        self.barynormals = get_barynormals(point_normals, simplices)
        self.areas = get_areas(points, simplices)
        self.owners, self.neighbours, self.edges_to_vertices = get_edge_connectivities(simplices)
        self.edge_lengths = get_edge_lengths(points, self.edges_to_vertices)
        self.edge_centers = get_edge_centers(points, self.edges_to_vertices)
        self.edge_tangents, self.edge_bitangents, self.edge_normals = get_edge_tbns(points, point_normals, self.edges_to_vertices)

        # Change the direction to make sure edge point from owners to neighbours
        directions = np.where(np.sum((self.edge_centers - self.barycenters[self.owners]) * self.edge_normals, axis=1) > 0, 1, -1)
        new_owners = np.where(directions == 1, self.owners, self.neighbours)
        new_neighbours = np.where(directions == 1, self.neighbours, self.owners)
        self.owners = new_owners
        self.neighbours = new_neighbours

        self.cells_to_edges = get_cell_connectivities_to_edge(self.owners, self.neighbours, len(self.areas))
        self.edge_weighing_factor = get_edge_weight_factors(points, point_normals, self.barycenters, self.owners, self.neighbours, self.edges_to_vertices)
        self.skewness, self.skewness_vector = get_skewness(points, point_normals, self.barycenters, self.owners, self.neighbours, self.edges_to_vertices)
        
        self.points = points
        self.point_normals = point_normals
        self.simplices = simplices

        # Some geometry operators
        # Compute cf, d_cf, and e
        self.cf = self.barycenters[self.neighbours] - self.barycenters[self.owners]
        self.d_cf = np.linalg.norm(self.cf, axis=1)
        self.e = np.squeeze(self.cf / self.d_cf[:, np.newaxis])

        # Compute ef
        self.ef = self.edge_lengths / np.sum(self.e * self.edge_normals, axis=1)

        # Construct owner and neighbour indices
        self.owner_indices = self.owners[self.neighbours != -1]
        self.neighbour_indices = self.neighbours[self.neighbours != -1]

        # Compute Tf
        self.normal_dot_e = np.sum(self.edge_normals * self.e, axis=1, keepdims=True)
        self.tf = (self.edge_normals - self.e / self.normal_dot_e) * self.edge_lengths[:, np.newaxis]

        self.diffusion_operator = {}
        self.convection_operator = {}
        self.possion_operator = {}

    def print_members(self):
        for name, value in self.__dict__.items():
            print(f"{name} = {value}")
    
    def interpolate_field_cell_to_face(self, phi):
        return (1.0-self.edge_weighing_factor[:, np.newaxis]) * phi[self.owners,...] + \
                 np.where(self.neighbours[:, np.newaxis] != -1,\
                self.edge_weighing_factor[:, np.newaxis] * phi[self.neighbours,...],np.zeros_like(phi[self.owners,...]))

    def reconstruct_surface_gradient(self, phi, n_skewness_corr_iter=1):
        # Init dphi_f
        dphi_f = np.zeros_like(self.edge_centers)
        dphi = np.zeros((len(self.areas),3))

        for _ in range(1+n_skewness_corr_iter):
            # Estimate phi_f
            phi_f = (1.0 - self.edge_weighing_factor) * phi[self.owners] + np.where(self.neighbours != -1,self.edge_weighing_factor * phi[self.neighbours],np.zeros_like(self.owners))

            # Correct phi_f
            phi_f += np.einsum('ij,ij->i', dphi_f, self.skewness_vector)

            # Estimate \nabla \phi_C
            dphi.fill(0)
            np.add.at(dphi, self.owner_indices, phi_f[:, np.newaxis] \
                * self.edge_lengths[:, np.newaxis] \
                * project_vectors_to_planes(self.edge_normals, self.barynormals[self.owner_indices], rotate=True) / self.areas[self.owner_indices, np.newaxis])
            np.add.at(dphi, self.neighbour_indices, -phi_f[:, np.newaxis] \
                * self.edge_lengths[:, np.newaxis] \
                * project_vectors_to_planes(self.edge_normals, self.barynormals[self.neighbour_indices], rotate=True) / self.areas[self.neighbour_indices, np.newaxis])

            # (here to enforce boundary condition)

            # Estimate phi_f
            dphi_f = (1.0 - self.edge_weighing_factor[:, np.newaxis]) * dphi[self.owners] + np.where(self.neighbours[:, np.newaxis] != -1, self.edge_weighing_factor[:, np.newaxis] * dphi[self.neighbours],np.zeros_like(dphi[self.owners]))
            
            # Enforce neumann boundary
            #for edge_ind, center, neighbour in zip(range(len(self.neighbours)), self.edge_centers, self.neighbours):
            #    if neighbour == -1:
            #        dphi_f[edge_ind] = 0.0
        
        return dphi_f, dphi
    
    def reconstruct_cell_gradient(self, phi):
        # Construct least-square operator
        dphi = np.zeros_like(self.barycenters)
        for cell_idx in range(len(self.areas)):
            LS = np.zeros((3,3))
            rhs = np.zeros((1,3))
            for face_idx in self.cells_to_edges[cell_idx]:
                nb_idx = self.neighbours[face_idx] if self.owners[face_idx] == cell_idx else self.owners[face_idx]
                # Fill the matrix
                cf = self.barycenters[nb_idx, :] - self.barycenters[cell_idx, :]
                cf = project_vector_to_plane(cf, self.barynormals[cell_idx])
                wk = 1.0 / np.sqrt(np.sum(cf**2))
                LS += wk * np.outer(cf, cf)
                dphi_k = phi[nb_idx] - phi[cell_idx]
                rhs += wk*cf*dphi_k

            # Solve the augmented matrix using least squares method
            dphi[cell_idx] = np.squeeze(np.linalg.lstsq(LS, rhs.T, rcond=None)[0])

            # Ensure that the solution satisfies the condition x⋅n = 0
            #dphi[cell_idx] = project_vector_to_plane(dphi[cell_idx], self.barynormals[cell_idx])

        return dphi
    
    def diffusion_matrix(self, phi, gamma, dirty=False):
        # Estimate phi_f
        gamma_f = 1.0 / (self.edge_weighing_factor / gamma[self.owners] + \
            np.where(self.neighbours != -1,(1.0 - self.edge_weighing_factor) / gamma[self.neighbours],np.zeros_like(self.owners)))

        if dirty or self.diffusion_operator == {}:
            print("Reset diffusion operator.")
            self.diffusion_operator = {}

            # Construct owner and neighbour indices
            owner_indices = self.owner_indices
            neighbour_indices = self.neighbour_indices

            # Compute data values
            val = gamma_f * self.ef / self.d_cf
            val = val[self.neighbours != -1]
            data_values = np.concatenate([val,-val,val,-val])
            
            # Compute row and column indices
            row_indices = np.concatenate([owner_indices, owner_indices, neighbour_indices, neighbour_indices])
            col_indices = np.concatenate([owner_indices, neighbour_indices, neighbour_indices, owner_indices])

            # Create sparse matrix
            Jf = coo_array((data_values,(row_indices,col_indices)), shape=(len(self.areas), len(self.areas)))
        
            # Construct rhs
            rhs = np.zeros(len(self.areas))

            self.diffusion_operator["row_indices"] = row_indices
            self.diffusion_operator["col_indices"] = col_indices
            self.diffusion_operator["data_values"] = data_values
            self.diffusion_operator["Jf"] = Jf
            self.diffusion_operator["rhs"] = rhs
        else:
            # Compute data values
            val = gamma_f * self.ef / self.d_cf
            val = val[self.neighbours != -1]
            np.concatenate([val,-val,val,-val], out=self.diffusion_operator["data_values"])
        
        # Update rhs
        dphi_f, _ = self.reconstruct_surface_gradient(phi)
        corr = np.sum((dphi_f * gamma_f[:, np.newaxis]) * self.tf, axis=1)
        self.diffusion_operator["rhs"].fill(0)
        np.add.at(self.diffusion_operator["rhs"], self.owner_indices, corr[self.owner_indices])
        np.add.at(self.diffusion_operator["rhs"], self.neighbour_indices, -corr[self.neighbour_indices])

        # Update sparse matrix
        self.diffusion_operator["Jf"].data = self.diffusion_operator["data_values"]

        return self.diffusion_operator["Jf"], self.diffusion_operator["rhs"]
    
    def convection_matrix(self, phi, rhou, quick=False, face_flux=False, dirty=False):
        valid_faces = self.neighbours != -1
        # Estimate phi_f
        if face_flux:
            rhou_f = rhou
        else:
            rhou_f = (1.0 - self.edge_weighing_factor[:, np.newaxis]) * rhou[self.owners] + \
                         np.where(self.neighbours[:, np.newaxis] != -1, self.edge_weighing_factor[:, np.newaxis] * rhou[self.neighbours], np.zeros_like(rhou[self.owners]))

        dphi_f, dphi = self.reconstruct_surface_gradient(phi)

        if dirty or self.convection_operator == {}:
            print("Reset convection operator.")
            self.convection_operator = {}

            # Construct owner and neighbour indices
            owner_indices = self.owner_indices
            neighbour_indices = self.neighbour_indices

            # Compute data values
            val = np.sum(rhou_f * self.edge_normals, axis=1) * self.edge_lengths
            val = val[valid_faces]
            val_p = val * (val > 0).astype(float)
            val_n = val * (val < 0).astype(float)
            data_values = np.concatenate([val_p,val_n,-val_n,-val_p])
            
            # Compute row and column indices
            row_indices = np.concatenate([owner_indices, owner_indices, neighbour_indices, neighbour_indices])
            col_indices = np.concatenate([owner_indices, neighbour_indices, neighbour_indices, owner_indices])

            # Create sparse matrix
            Jf = coo_array((data_values,(row_indices,col_indices)), shape=(len(self.areas), len(self.areas)))
        
            # Construct rhs
            rhs = np.zeros(len(self.areas))

            self.convection_operator["row_indices"] = row_indices
            self.convection_operator["col_indices"] = col_indices
            self.convection_operator["data_values"] = data_values
            self.convection_operator["Jf"] = Jf
            self.convection_operator["rhs"] = rhs
        else:
            # Compute data values
            val = np.sum(rhou_f * self.edge_normals, axis=1) * self.edge_lengths
            val = val[valid_faces]
            val_p = val * (val > 0).astype(float)
            val_n = val * (val < 0).astype(float)
            np.concatenate([val_p,val_n,-val_n,-val_p], out=self.convection_operator["data_values"])

            self.convection_operator["rhs"].fill(0)

        if quick:
            np.add.at(self.convection_operator["rhs"], self.owner_indices,     -val_p * 0.5 * np.sum((dphi[self.owner_indices]+dphi_f[valid_faces]) * (self.edge_centers[valid_faces] - self.barycenters[self.owner_indices]), axis=1))
            np.add.at(self.convection_operator["rhs"], self.neighbour_indices,  val_p * 0.5 * np.sum((dphi[self.owner_indices]+dphi_f[valid_faces]) * (self.edge_centers[valid_faces] - self.barycenters[self.owner_indices]), axis=1))
            np.add.at(self.convection_operator["rhs"], self.owner_indices,     -val_n * 0.5 * np.sum((dphi[self.neighbour_indices]+dphi_f[valid_faces]) * (self.edge_centers[valid_faces] - self.barycenters[self.neighbour_indices]), axis=1))
            np.add.at(self.convection_operator["rhs"], self.neighbour_indices,  val_n * 0.5 * np.sum((dphi[self.neighbour_indices]+dphi_f[valid_faces]) * (self.edge_centers[valid_faces] - self.barycenters[self.neighbour_indices]), axis=1))
        
        # Update sparse matrix
        self.convection_operator["Jf"].data = self.convection_operator["data_values"]

        return self.convection_operator["Jf"], self.convection_operator["rhs"]
    
    def identity_matrix(self):
        # Construct identity operator
        row, col, data = [], [], []
        for cell_idx in range(len(self.areas)):
            row.append(cell_idx)
            col.append(cell_idx)
            data.append(self.areas[cell_idx])

        Iv = coo_array((data, (row, col)), shape=(len(self.areas), len(self.areas)))

        return Iv

    def divergence(self, uf):
        div = np.zeros((len(self.areas),))
        for fi, normal, center, sf, owner, neighbour in zip(uf, self.edge_normals, self.edge_centers, self.edge_lengths, self.owners, self.neighbours):
            div[owner] += fi * sf
            if neighbour != -1:
                div[neighbour] -= fi * sf
        return np.sum(np.abs(div)), np.max(div), np.min(div)

    def helmholtz_projection(self, mass_flow_rate, dirty=False):
        valid_faces = self.neighbours != -1
        # Estimate phi_f
        #rhou_f = (1.0 - self.edge_weighing_factor[:, np.newaxis]) * rhou[self.owners] + \
        #                 np.where(self.neighbours[:, np.newaxis] != -1,self.edge_weighing_factor[:, np.newaxis] * rhou[self.neighbours],np.zeros_like(rhou[self.owners]))
        #mass_flow = mass_flow_rate

        if dirty or self.possion_operator == {}:
            print("Reset possion operator.")
            self.possion_operator = {}

            # Construct owner and neighbour indices
            owner_indices = self.owner_indices
            neighbour_indices = self.neighbour_indices

            # Compute data values
            val = self.ef / self.d_cf
            val = val[valid_faces]
            data_values = np.concatenate([val,-val,val,-val])
            
            # Compute row and column indices
            row_indices = np.concatenate([owner_indices, owner_indices, neighbour_indices, neighbour_indices])
            col_indices = np.concatenate([owner_indices, neighbour_indices, neighbour_indices, owner_indices])

            # Create sparse matrix
            Jf = coo_array((data_values,(row_indices,col_indices)), shape=(len(self.areas), len(self.areas)))
        
            # Construct rhs
            rhs = np.zeros(len(self.areas))

            # Construct dphi_fn
            dphi_fn = 0*self.edge_lengths

            self.possion_operator["row_indices"] = row_indices
            self.possion_operator["col_indices"] = col_indices
            self.possion_operator["data_values"] = data_values
            self.possion_operator["Jf"] = Jf
            self.possion_operator["rhs"] = rhs
            self.possion_operator["dphi_fn"] = dphi_fn

        # Update rhs
        self.possion_operator["rhs"].fill(0)
        np.add.at(self.possion_operator["rhs"], self.owner_indices, mass_flow_rate[valid_faces] * self.edge_lengths[valid_faces])
        np.add.at(self.possion_operator["rhs"], self.neighbour_indices, -mass_flow_rate[valid_faces] * self.edge_lengths[valid_faces])

        phi = spla.gmres(self.possion_operator["Jf"], self.possion_operator["rhs"], tol=1e-6)[0]

        # Reconstruct surface normal gradients
        self.possion_operator["dphi_fn"].fill(0)
        self.possion_operator["dphi_fn"][valid_faces] = (phi[self.neighbour_indices] - phi[self.owner_indices]) / self.d_cf[valid_faces] / np.squeeze(self.normal_dot_e)[valid_faces]

        # Correct surface flux
        new_mass_flow_rate = mass_flow_rate + self.possion_operator["dphi_fn"]

        _, dphi = self.reconstruct_surface_gradient(phi)
        #dphi = self.reconstruct_cell_gradient(phi)

        return new_mass_flow_rate, dphi
    
    def get_total_kinetic_emergy(self, u):
        E = 0
        for cell_idx in range(len(self.areas)):
            E += np.sum(u[cell_idx,:]**2) * self.areas[cell_idx]
        return E
    