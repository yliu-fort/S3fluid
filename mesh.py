import numpy as np
from scipy.spatial import Delaunay, minkowski_distance
import meshio
from icosphere import icosphere
from scipy.sparse import coo_array, csc_matrix
import scipy.sparse.linalg as spla
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
        self.edge_weighing_factor = get_edge_weight_factors(points, point_normals, self.barycenters, self.owners, self.neighbours, self.edges_to_vertices)
        self.skewness, self.skewness_vector = get_skewness(points, point_normals, self.barycenters, self.owners, self.neighbours, self.edges_to_vertices)
        
        self.points = points
        self.point_normals = point_normals
        self.simplices = simplices

    def interpolate_field_cell_to_face(phi):
        return self.edge_weighing_factor[:, np.newaxis] * phi[self.owners] + \
                 np.where(self.neighbours != -1,\
                (1.0 - self.edge_weighing_factor[:, np.newaxis]) * phi[self.neighbours],np.zeros_like(phi[self.owners]))

    def reconstruct_surface_gradient(self, phi, n_skewness_corr_iter=1):
        # Init dphi_f
        dphi_f = np.zeros_like(self.edge_centers)

        for _ in range(1+n_skewness_corr_iter):
            # Estimate phi_f
            phi_f = self.edge_weighing_factor * phi[self.owners] + np.where(self.neighbours != -1,(1.0 - self.edge_weighing_factor) * phi[self.neighbours],np.zeros_like(self.owners))

            # Correct phi_f
            phi_f += np.einsum('ij,ij->i', dphi_f, self.skewness_vector)

            # Estimate \nabla \phi_C
            dphi = np.zeros((len(self.areas),3))
            for phi_fi, normal, center, sf, owner, neighbour in zip(phi_f, self.edge_normals, self.edge_centers, self.edge_lengths, self.owners, self.neighbours):
                e = center - self.barycenters[owner] # (todo) need revision
                direction = 1 if np.dot(e, normal) > 0 else -1
                dphi[owner] += phi_fi * sf * normal / self.areas[owner] * direction
                if neighbour != -1:
                    dphi[neighbour] -= phi_fi * sf * normal / self.areas[neighbour] * direction
                    
            # (here to enforce boundary condition)

            # Estimate phi_f
            dphi_f = self.edge_weighing_factor[:, np.newaxis] * dphi[self.owners] + np.where(self.neighbours != -1, (1.0 - self.edge_weighing_factor[:, np.newaxis]) * dphi[self.neighbours],np.zeros_like(dphi[self.owners]))
            
            # Enforce neumann boundary
            for edge_ind, center, neighbour in zip(range(len(self.neighbours)), self.edge_centers, self.neighbours):
                if neighbour == -1:
                    dphi_f[edge_ind] = 0.0
        
        return dphi_f, dphi
    
    def diffusion_matrix(self, phi, gamma):
        # Estimate phi_f
        gamma_f = 1.0 / (self.edge_weighing_factor / gamma[self.owners] + \
                         np.where(self.neighbours != -1,(1.0 - self.edge_weighing_factor) / gamma[self.neighbours],np.zeros_like(self.owners)))
        
        dphi_f, _ = self.reconstruct_surface_gradient(phi)

        # Construct diffusion operator
        row, col, data = [], [], []
        rhs = np.zeros((len(self.areas),))
        for gamma_fi, dphi_fi, normal, center, sf, owner, neighbour in zip(gamma_f, dphi_f, self.edge_normals, self.edge_centers, self.edge_lengths, self.owners, self.neighbours):
                direction = 1 if np.dot(center - self.barycenters[owner], normal) > 0 else -1
                if neighbour != -1:
                    cf = self.barycenters[neighbour] - self.barycenters[owner]
                    d_cf = np.linalg.norm(cf)
                    e = cf/d_cf
                    ef = sf/np.dot(e, normal*direction)

                    row.append(owner)
                    col.append(owner)
                    data.append(gamma_fi*ef/d_cf)
                    row.append(owner)
                    col.append(neighbour)
                    data.append(-gamma_fi*ef/d_cf)
                    row.append(neighbour)
                    col.append(neighbour)
                    data.append(gamma_fi*ef/d_cf)
                    row.append(neighbour)
                    col.append(owner)
                    data.append(-gamma_fi*ef/d_cf)

                    rhs[owner] += np.dot(dphi_fi*gamma_fi, normal*direction-e/np.dot(e, normal*direction)) * sf
                    rhs[neighbour] -= np.dot(dphi_fi*gamma_fi, normal*direction-e/np.dot(e, normal*direction)) * sf

        Jf = coo_array((data, (row, col)), shape=(len(self.areas), len(self.areas)))

        return Jf, rhs
    
    def convection_matrix(self, phi, rhou, quick=False, face_flux=False):
        # Estimate phi_f
        if face_flux:
            rhou_f = rhou
        else:
            rhou_f = self.edge_weighing_factor[:, np.newaxis] * rhou[self.owners] + \
                         np.where(self.neighbours != -1, (1.0 - self.edge_weighing_factor[:, np.newaxis]) * rhou[self.neighbours], np.zeros_like(rhou[self.owners]))

        dphi_f, dphi = self.reconstruct_surface_gradient(phi)

        # Construct diffusion operator
        row, col, data = [], [], []
        rhs = np.zeros((len(self.areas),))
        for rhou_fi, dphi_fi, normal, center, sf, owner, neighbour in zip(rhou_f, dphi_f, self.edge_normals, self.edge_centers, self.edge_lengths, self.owners, self.neighbours):
                direction = 1 if np.dot(center - self.barycenters[owner], normal) > 0 else -1
                if neighbour != -1:
                    convection_direction = 1 if np.dot(rhou_fi, normal*direction) > 0 else -1
                    
                    # Upwind \phi_f = \phi_C
                    a = np.dot(rhou_fi, normal * direction) * sf
                    if convection_direction > 0:
                        row.append(owner)
                        col.append(owner)
                        data.append(a)
                        row.append(neighbour)
                        col.append(owner)
                        data.append(-a)
                    else:
                        row.append(owner)
                        col.append(neighbour)
                        data.append(a)
                        row.append(neighbour)
                        col.append(neighbour)
                        data.append(-a)

                    if quick:
                        # QUICK high-order terms
                        if convection_direction > 0:
                            cf = center - self.barycenters[owner]
                            rhs[owner] -= a * 0.5 * np.dot(dphi[owner]+dphi_fi, cf)
                            rhs[neighbour] += a * 0.5 * np.dot(dphi[owner]+dphi_fi, cf)
                        else:
                            cf = center - self.barycenters[neighbour]
                            rhs[owner] -= a * 0.5 * np.dot(dphi[neighbour]+dphi_fi, cf)
                            rhs[neighbour] += a * 0.5 * np.dot(dphi[neighbour]+dphi_fi, cf)

                else:
                    convection_direction = 1 if np.dot(rhou_fi, normal*direction) > 0 else -1
                    # Upwind \phi_f = \phi_C
                    a = np.dot(rhou_fi, normal * direction) * sf
                    if convection_direction > 0:
                        row.append(owner)
                        col.append(owner)
                        data.append(a)
                    else:
                        if center[0] < 1e-4:
                            rhs[owner] -= a * 1.0 # \phi_B = 1.0
                        elif center[1] < 1e-4:
                            rhs[owner] -= a * 0.0 # \phi_B = 0.0

        Jf = coo_array((data, (row, col)), shape=(len(self.areas), len(self.areas)))

        return Jf, rhs
    
    def identity_matrix(self):
        # Construct identity operator
        row, col, data = [], [], []
        for cell_idx in range(len(self.areas)):
            row.append(cell_idx)
            col.append(cell_idx)
            data.append(self.areas[cell_idx])

        Iv = coo_array((data, (row, col)), shape=(len(self.areas), len(self.areas)))

        return Iv

    def helmholtz_projection(self, mass_flow_rate, sngrad_corr=False):
        # Estimate phi_f
        #rhou_f = self.edge_weighing_factor[:, np.newaxis] * rhou[self.owners] + \
        #                 np.where(self.neighbours != -1,(1.0 - self.edge_weighing_factor[:, np.newaxis]) * rhou[self.neighbours],np.zeros_like(rhou[self.owners]))
        #mass_flow = mass_flow_rate

        # Construct decomposition operator
        row, col, data = [], [], []
        rhs = np.zeros((len(self.areas),))
        for dphi_fi, normal, center, sf, owner, neighbour in zip(mass_flow_rate, self.edge_normals, self.edge_centers, self.edge_lengths, self.owners, self.neighbours):
                direction = 1 if np.dot(center - self.barycenters[owner], normal) > 0 else -1
                if neighbour != -1:
                    cf = self.barycenters[neighbour] - self.barycenters[owner]
                    d_cf = np.linalg.norm(cf)
                    e = cf/d_cf
                    ef = sf/np.dot(e, normal*direction)

                    row.append(owner)
                    col.append(owner)
                    data.append(ef/d_cf)
                    row.append(owner)
                    col.append(neighbour)
                    data.append(-ef/d_cf)
                    row.append(neighbour)
                    col.append(neighbour)
                    data.append(ef/d_cf)
                    row.append(neighbour)
                    col.append(owner)
                    data.append(-ef/d_cf)

                    rhs[owner] += dphi_fi * sf * direction
                    rhs[neighbour] -= dphi_fi * sf * direction

        Jf = coo_array((data, (row, col)), shape=(len(self.areas), len(self.areas)))

        phi = spla.gmres(Jf, rhs, tol=1e-7)[0]

        if sngrad_corr:
            dphi_f, _ = self.reconstruct_surface_gradient(phi)
            dphi_fn = np.einsum("ij, ij->i", dphi_f, self.edge_normals)
        else:
            dphi_fn = 0*self.edge_lengths
            for face_idx, normal, center, sf, owner, neighbour in zip(range(len(self.edge_lengths)), self.edge_normals, self.edge_centers, self.edge_lengths, self.owners, self.neighbours):
                    direction = 1 if np.dot(center - self.barycenters[owner], normal) > 0 else -1
                    if neighbour != -1:
                        cf = self.barycenters[neighbour] - self.barycenters[owner]
                        d_cf = np.linalg.norm(cf)
                        e = cf/d_cf

                        dphi_fn[face_idx] = (phi[neighbour] - phi[owner]) / d_cf / np.dot(e, normal)

        # Reconstruct surface normal gradient
        new_mass_flow_rate = mass_flow_rate + dphi_fn

        div = np.zeros((len(self.areas),))
        for dphi_fi, normal, center, sf, owner, neighbour in zip(new_mass_flow_rate, self.edge_normals, self.edge_centers, self.edge_lengths, self.owners, self.neighbours):
            direction = 1 if np.dot(center - self.barycenters[owner], normal) > 0 else -1
            div[owner] += dphi_fi * sf * direction
            if neighbour != -1:
                div[neighbour] -= dphi_fi * sf * direction

        return new_mass_flow_rate, div