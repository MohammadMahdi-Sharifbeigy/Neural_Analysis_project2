import numpy as np

def angleVec(a, b):
    """
    MATLAB: function ang = angleVec(a,b)
    
    Calculates the angle (in degrees) between vectors a and b, row-wise.
    a and b must be NumPy arrays with the same number of rows and columns.
    
    Args:
        a (np.ndarray): The first vector matrix.
        b (np.ndarray): The second vector matrix.
        
    Returns:
        np.ndarray: A 1D array of angles in degrees.
    """
    
    # Ensure a and b are NumPy arrays
    a = np.array(a)
    b = np.array(b)
    
    if a.shape != b.shape:
        raise ValueError("Inputs 'a' and 'b' must have the same shape.")
        
    num_vectors = a.shape[0]
    ang = np.zeros(num_vectors)
    
    for i in range(num_vectors):
        # dot product of the i-th row (vector)
        dot_ab = np.dot(a[i,:], b[i,:])
        
        # Norms squared (dot(v,v))
        norm_a_sq = np.dot(a[i,:], a[i,:])
        norm_b_sq = np.dot(b[i,:], b[i,:])
        
        # Calculate the argument for arccos: cos(theta) = (a . b) / (||a|| * ||b||)
        # Add a small epsilon to the denominator to prevent division by zero, 
        # as a perfect zero-norm vector argument may indicate the angle is undefined or 0.
        denominator = np.sqrt(norm_a_sq * norm_b_sq)
        
        # Handle zero vector case (optional, but safer)
        if denominator == 0:
             # If either vector is zero, the angle is often considered undefined or 0.
             # We set it to 0 here to avoid division by zero.
             angle_rad = 0.0
        else:
            # Clamp the argument to the valid range [-1, 1] for acos 
            # due to potential floating-point errors
            cos_theta = dot_ab / denominator
            cos_theta = np.clip(cos_theta, -1.0, 1.0)
            
            # Calculate angle in radians
            angle_rad = np.arccos(cos_theta)
            
        # Convert to degrees and store (MATLAB: *180/pi)
        ang[i] = angle_rad * 180.0 / np.pi
        
    return ang