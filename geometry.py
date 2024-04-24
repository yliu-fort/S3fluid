# %%
import numpy as np
from scipy.spatial import minkowski_distance

def normalize(vector):
    """Normalize a vector.

    Args:
        vector (np.ndarray): A numpy array of vectors to normalize.

    Returns:
        np.ndarray: A numpy array of normalized vectors.

    Raises:
        ValueError: If the input is not a 2D numpy array or if any vector has zero length.
    """
    if not isinstance(vector, np.ndarray) or len(vector.shape) != 2:
        raise ValueError("Input should be a 2D numpy array")
    norms = np.linalg.norm(vector, axis=0)
    if np.any(norms == 0):
        raise ValueError("Zero length vector detected")
    return vector / norms

def get_barycenters(points, simplices):
    """Calculate the barycenters of simplices defined by points.

    Args:
        points (np.ndarray): An array of point coordinates.
        simplices (list of list of int): Indices of vertices forming each simplex.

    Returns:
        np.ndarray: Coordinates of the barycenters of the simplices.
    """
    # Calculate the mean of points indexed by simplices
    barycenters = np.array([np.mean(points[simplice], axis=0) for simplice in simplices])
    return barycenters.T

def get_barynormals(point_normals, simplices):
    """Calculate the barycentric normals for given simplices.

    Args:
        point_normals (np.ndarray): An array of normals at each point.
        simplices (list of lists): A list of simplices, where each simplex
                                   is defined by indices into point_normals.

    Returns:
        np.ndarray: An array of the mean normals for each simplex, transposed.
    """
    mean_normals = [
        np.mean(point_normals[simplex], axis=0)
        for simplex in simplices
    ]
    return np.array(mean_normals).T

import numpy as np

def get_edge_connectivities(simplices):
    """Generate edge connectivities for a given set of simplices.

    Args:
        simplices (list of list of int): Each sublist represents a simplex with vertex indices.

    Returns:
        tuple: Three NumPy arrays containing owners, neighbours of edges, and the edges as vertex pairs.
    """
    # Dictionary to keep track of edges and their associated simplices
    edges = {}
    for simplex_index, simplex in enumerate(simplices):
        num_vertices = len(simplex)
        for i in range(num_vertices):
            # Create a sorted tuple to identify edges uniquely
            edge = tuple(sorted((simplex[i], simplex[(i + 1) % num_vertices])))
            if edge not in edges:
                edges[edge] = []
            edges[edge].append(simplex_index)

    edges_to_vertices = []
    owners = []
    neighbours = []

    # Process collected edges to determine ownership and neighboring simplices
    for edge, linked_simplices in edges.items():
        edges_to_vertices.append(edge)
        owners.append(linked_simplices[0])
        # Assign -1 if there is no neighbor simplex
        neighbours.append(-1 if len(linked_simplices) == 1 else linked_simplices[1])

    return np.array(owners), np.array(neighbours), np.array(edges_to_vertices)

def get_cell_connectivities_to_edge(owners, neighbours, ncells):
    """Generates mappings from cells to edges based on ownership and neighborhood.

    Args:
        owners (list of int): List of indices of the owning cells for each face.
        neighbours (list of int): List of indices of the neighboring cells for each face.
        ncells (int): Total number of cells.

    Returns:
        list of list of int: Each cell index maps to a list of edge indices connected to it.
    """
    # Initialize list of lists to store edge connectivities for each cell
    cells_to_edges = [[] for _ in range(ncells)]
    
    # Populate the list with edges each cell is connected to
    for face_idx, owner, neighbour in zip(range(len(owners)), owners, neighbours):
        cells_to_edges[owner].append(face_idx)
        cells_to_edges[neighbour].append(face_idx)
    
    return cells_to_edges

def get_edge_lengths(points, edges_to_vertices):
    # Compute edge length (face area), edge (face) center and centroid of edges (faces)
    return minkowski_distance(points[[x[0] for x in edges_to_vertices]],points[[x[1] for x in edges_to_vertices]])

def get_edge_centers(points, edges_to_vertices):
    return np.array([np.mean(np.array([points[v0] for v0 in v]),axis=0) for v in edges_to_vertices]).T

def get_edge_tbns(points, point_normals, edges_to_vertices):
    """Computes the tangent, bitangent, and normal vectors for edges.

    Args:
        points (ndarray): The coordinates of the points.
        point_normals (ndarray): The normals at each point.
        edges_to_vertices (list of tuple): Each tuple contains the indices of the vertices that form an edge.

    Returns:
        tuple: Three ndarrays representing the tangents, bitangents, and normals of the edges.
    """
    # Calculate the lengths of each edge.
    edge_lengths = get_edge_lengths(points, edges_to_vertices)

    # Calculate edge tangents.
    edge_tangents = np.array([
        (points[v[1]] - points[v[0]]) / l for v, l in zip(edges_to_vertices, edge_lengths)
    ])

    # Calculate edge bitangents.
    edge_bitangents = np.array([
        np.mean(np.array([point_normals[v0] for v0 in v]), axis=0) for v in edges_to_vertices
    ])

    # Calculate edge normals.
    edge_normals = np.array([
        np.cross(t, r) for t, r in zip(edge_tangents, edge_bitangents)
    ])

    return edge_tangents.T, edge_bitangents.T, edge_normals.T

def get_areas(points, simplices):
    def herons_formula(a, b, c):
        """ Calculate the area of a triangle given the lengths of its sides using Heron's formula. """
        s = (a + b + c) / 2
        return np.sqrt(s * (s - a) * (s - b) * (s - c))

    def area_of_polygon(points, vertices):
        """ Calculate the area of a polygon by triangulating it into multiple triangles. """
        if len(vertices) == 3:  # It's a triangle
            a = minkowski_distance(points[vertices[0]], points[vertices[1]])
            b = minkowski_distance(points[vertices[1]], points[vertices[2]])
            c = minkowski_distance(points[vertices[2]], points[vertices[0]])
            return herons_formula(a, b, c)
        else:  # It's a polygon with more than 3 vertices, triangulate it
            area = 0
            for i in range(1, len(vertices) - 1):
                a = minkowski_distance(points[vertices[0]], points[vertices[i]])
                b = minkowski_distance(points[vertices[i]], points[vertices[i + 1]])
                c = minkowski_distance(points[vertices[i + 1]], points[vertices[0]])
                area += herons_formula(a, b, c)
            return area

    # Calculate the area for each simplex
    return np.array([area_of_polygon(points, simplex) for simplex in simplices])

def get_edge_weight_factors(points, point_normals, barycenters, owners, neighbours, edges_to_vertices):
    # Edge (Face) weighing factor
    edge_weighing_factor = []

    def point_to_line_distance(point, x0, x1):
        """ Calculate the normal distance from a point to a line defined by two points (x0 and x1) in 3D,
            handling edge cases such as degenerate lines. """
        # Convert points to numpy arrays if not already
        point = np.asarray(point)
        x0 = np.asarray(x0)
        x1 = np.asarray(x1)

        # Calculate the direction vector of the line
        d = x1 - x0

        # Check if the direction vector is zero (degenerate line segment)
        if np.allclose(d, 0):
            print("Warning: The input points for the line are the same. The line is degenerate.")
            return np.linalg.norm(point - x0)

        # Calculate the vector from x0 to the point
        p = point - x0

        # Calculate the cross product of d and p
        c = np.cross(d, p)

        # Calculate the distance from the point to the line (magnitude of cross product divided by magnitude of d)
        distance = np.linalg.norm(c) / np.linalg.norm(d)
        
        return distance

    for owner, neighbour, vertices in zip(owners, neighbours, edges_to_vertices):
        distance_to_owner = point_to_line_distance(barycenters[:,owner], points[vertices[0]], points[vertices[1]])
        distance_to_neighbour = point_to_line_distance(barycenters[:,neighbour], points[vertices[0]], points[vertices[1]]) \
            if neighbour != -1 else 0
        edge_weighing_factor.append(distance_to_owner / (distance_to_owner + distance_to_neighbour))

    return np.array(edge_weighing_factor)

def get_skewness(points, point_normals, barycenters, owners, neighbours, edges_to_vertices):
    # Skewness
    def project_point_onto_plane(v, p, n):
        n = np.array(n)
        v = np.array(v)
        p = np.array(p)
        return v - np.dot(v - p, n) / np.dot(n, n) * n

    def find_intersection(v1, v2, p, t, n):
        v1_proj = project_point_onto_plane(v1, p, n)
        v2_proj = project_point_onto_plane(v2, p, n)
        d1 = v2_proj - v1_proj
        t = np.array(t)

        # Matrix to solve for parameters t (for the projected line) and s (for the line in the plane)
        A = np.column_stack([d1, -t])
        b = np.array(p) - v1_proj

        # Check if the matrix A is singular (i.e., lines are parallel)
        if np.linalg.matrix_rank(A) < 2:
            return None  # Lines are parallel or coincident; no unique intersection

        # Solve the linear system
        params, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
        if residuals.size > 0 and not np.isclose(residuals, 0):
            return None  # The lines do not intersect

        intersection_point = v1_proj + params[0] * d1
        return intersection_point

    skewness = []
    skewness_vector = []
    edge_lengths = get_edge_lengths(points, edges_to_vertices)
    edge_centers = get_edge_centers(points, edges_to_vertices)
    edge_tangents, edge_bitangents, _ = get_edge_tbns(points, point_normals, edges_to_vertices)
    for owner, neighbour, center, length, tangent, bitangent in zip(owners, neighbours, edge_centers.T, edge_lengths, edge_tangents.T, edge_bitangents.T):
        intersection = find_intersection(barycenters[:, owner], barycenters[:, neighbour], center, tangent, bitangent) if neighbour != -1 else None
        distance = minkowski_distance(intersection, center) if neighbour != -1 else 0
        skewness.append(2.0 * distance / length)
        skewness_vector.append(center - intersection if neighbour != -1 else center - center)

    skewness = np.array(skewness)
    skewness_vector = np.array(skewness_vector).T
    return skewness, skewness_vector

def project_vector_to_plane(v, n):
    # Convert v and n into numpy arrays
    v = np.array(v)
    n = np.array(n)
    
    # Check if the normal vector is zero
    if np.all(n == 0):
        raise ValueError("The normal vector n must not be the zero vector.")

    # Normalize the normal vector to avoid division by a very small number
    n_norm = normalize(n)
    
    # Calculate the dot product of v and n
    dot_product = np.dot(v, n_norm)
    
    # Calculate the projection of v onto n
    projection_v_on_n = dot_product * n_norm
    
    # Calculate the projection of v onto the plane
    projection_on_plane = v - projection_v_on_n
    
    return projection_on_plane

def project_vectors_to_planes(v, n, rotate=False):    
    # Calculate the projection of v onto the plane
    if rotate:
        vv = np.sum(v * v, axis=0)
        projected_v = v - np.sum(v * n, axis=0) * n
        projected_vv = np.sum(projected_v * projected_v, axis=0)
        return projected_v / projected_vv * vv
    else:
        return v - np.sum(v * n, axis=0) * n
    
if __name__=="__main__":
    np.random.seed(42)

    assert normalize(np.random.rand(3,7)).shape == (3,7)

    assert get_barycenters(np.array([[0, 0, 0], [0, 1, 0], [1, 0, 0], [1, 1, 0]]),[[2,3,0],[3,1,0]]).shape == (3,2)

    assert np.all(project_vectors_to_planes(np.array([[1,-1],[1,-1],[1,-1]]),np.array([[0,0],[0,0],[1,1]])) == np.array([[1,-1],[1,-1],[0,0]]))