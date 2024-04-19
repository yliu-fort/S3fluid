import numpy as np

def project_vector_to_plane(v, n):
    # Convert v and n into numpy arrays
    v = np.array(v)
    n = np.array(n)
    
    # Check if the normal vector is zero
    if np.all(n == 0):
        raise ValueError("The normal vector n must not be the zero vector.")

    # Normalize the normal vector to avoid division by a very small number
    n_norm = n / np.linalg.norm(n)
    
    # Calculate the dot product of v and n
    dot_product = np.dot(v, n_norm)
    
    # Calculate the projection of v onto n
    projection_v_on_n = dot_product * n_norm
    
    # Calculate the projection of v onto the plane
    projection_on_plane = v - projection_v_on_n
    
    return projection_on_plane

# Example usage:
v = [3, 4, 5]  # Example vector
n = [0, 0, 1]  # Normal vector of the plane

try:
    result = project_vector_to_plane(v, n)
    print("Projection of vector v onto the plane:", result)
except ValueError as e:
    print(e)
