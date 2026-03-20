import numpy as np

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

def test_find_intersection():
    # Test 1: Simple intersection
    v1 = [0, 0, 0]
    v2 = [1, 1, 0]
    p = [0, 0, 0]
    t = [1, 0, 0]
    n = [0, 0, 1]
    assert np.allclose(find_intersection(v1, v2, p, t, n), np.array([0, 0, 0])), "Test 1 Failed"

    # Test 2: Intersection at a specific point
    v1 = [1, 0, -1]
    v2 = [1, 0, 1]
    p = [0, 0, 0]
    t = [0, 1, 0]
    n = [1, 0, 0]
    assert np.allclose(find_intersection(v1, v2, p, t, n), np.array([0, 0, 0])), "Test 2 Failed"

    # Test 3: Parallel lines, no intersection
    v1 = [0, 1, 0]
    v2 = [1, 1, 0]
    p = [0, 0, 0]
    t = [1, 0, 0]
    n = [0, 1, 0]
    assert find_intersection(v1, v2, p, t, n) is None, "Test 3 Failed"

    # Test 4: Coplanar non-intersecting lines
    v1 = [0, 0, 0]
    v2 = [0, 1, 0]
    p = [0, 0, 1]
    t = [0, 1, 0]
    n = [1, 0, 0]
    assert find_intersection(v1, v2, p, t, n) is None, "Test 4 Failed"

    # Test 5: Collinear lines
    v1 = [0, 0, 0]
    v2 = [0, 2, 0]
    p = [0, 1, 0]
    t = [0, 1, 0]
    n = [1, 0, 0]
    assert find_intersection(v1, v2, p, t, n) is None, "Test 5 Failed"

    print("All tests passed!")

# Run the tests
test_find_intersection()
