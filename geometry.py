# %%
import numpy as np
from scipy.spatial import minkowski_distance

def normalize(v):
    # Normalize the normal vector
    return v / np.linalg.norm(v)

def get_barycenters(points, simplices):
    return np.array([np.mean(points[simplice],axis=0) for simplice in simplices])

def get_barynormals(point_normals, simplices):
    return np.array([np.mean(point_normals[simplice],axis=0) for simplice in simplices])

def get_edge_connectivities(simplices):
    # Generate edge list and edge owner & neighbor list
    edges = {}
    for simplex_index, simplex in enumerate(simplices):
        for i in range(len(simplex)):
            # Create sorted tuple to identify edges uniquely
            edge = tuple(sorted((simplex[i], simplex[(i + 1) % len(simplex)])))
            if edge not in edges:
                edges[edge] = []
            edges[edge].append(simplex_index)

    edges_to_vertices = []
    owners = []
    neighbours = []
    for edge, _simplices in edges.items():
        edges_to_vertices.append(edge)
        owners.append(_simplices[0])
        neighbours.append(-1 if len(_simplices) == 1 else _simplices[1] )
    return np.array(owners), np.array(neighbours), edges_to_vertices

def get_cell_connectivities_to_edge(owners, neighbours, ncells):
    # Generate edge list and edge owner & neighbor list
    cells_to_edges = [[] for _ in range(ncells)]
    for face_idx, owner, neighbour in zip(range(len(owners)), owners, neighbours):
        cells_to_edges[owner].append(face_idx)
        cells_to_edges[neighbour].append(face_idx)
        
    return cells_to_edges

def get_edge_lengths(points, edges_to_vertices):
    # Compute edge length (face area), edge (face) center and centroid of edges (faces)
    return minkowski_distance(points[[x[0] for x in edges_to_vertices]],points[[x[1] for x in edges_to_vertices]])

def get_edge_centers(points, edges_to_vertices):
    return np.array([np.mean(np.array([points[v0] for v0 in v]),axis=0) for v in edges_to_vertices])

def get_edge_tbns(points, point_normals, edges_to_vertices):
    # Compute edge tangent, co-tangent and normals
    edge_lengths = get_edge_lengths(points, edges_to_vertices)
    edge_tangents = np.array([(points[v[1]] - points[v[0]])/l for v, l in zip(edges_to_vertices, edge_lengths)])
    edge_bitangents = np.array([np.mean(np.array([point_normals[v0] for v0 in v]),axis=0) for v in edges_to_vertices])
    edge_normals = np.array([np.cross(t, r) for t, r in zip(edge_tangents, edge_bitangents)])
    return edge_tangents, edge_bitangents, edge_normals

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
        distance_to_owner = point_to_line_distance(barycenters[owner], points[vertices[0]], points[vertices[1]])
        distance_to_neighbour = point_to_line_distance(barycenters[neighbour], points[vertices[0]], points[vertices[1]]) \
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
    edge_tangents, edge_bitangents, edge_normals = get_edge_tbns(points, point_normals, edges_to_vertices)
    for owner, neighbour, center, length, tangent, bitangent in zip(owners, neighbours, edge_centers, edge_lengths, edge_tangents, edge_bitangents):
        intersection = find_intersection(barycenters[owner], barycenters[neighbour], center, tangent, bitangent) if neighbour != -1 else None
        distance = minkowski_distance(intersection, center) if neighbour != -1 else 0
        skewness.append(2.0 * distance / length)
        skewness_vector.append(center - intersection if neighbour != -1 else center - center)

    skewness = np.array(skewness)
    skewness_vector = np.array(skewness_vector)
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
        vv = np.sum(v * v, axis=1, keepdims=True)
        projected_v = v - np.sum(v*n, axis=1,keepdims=True) * n
        projected_vv = np.sum(projected_v * projected_v, axis=1, keepdims=True)
        return projected_v / projected_vv * vv
    else:
        return v - np.sum(v*n, axis=1,keepdims=True) * n