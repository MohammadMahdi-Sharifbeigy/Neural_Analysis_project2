import numpy as np

# We'll need the 'vertical' function for correct output shape
# from .vertical import vertical

def proj(a, b):
    """
    MATLAB: function p = proj(a,b)
    
    Calculates the vector projection of 'a' onto 'b'.
    Formula: p = (a . b) / (b . b) * b
    
    Args:
        a (np.ndarray): Vector a.
        b (np.ndarray): Vector b (onto which a is projected).
        
    Returns:
        np.ndarray: The projection vector p, respecting the column/row orientation of 'a'.
    """
    
    a = np.array(a).flatten()
    b = np.array(b).flatten()
    
    # The dot product needs to be calculated on 1D/flattened arrays.
    dot_ab = np.dot(a, b)
    dot_bb = np.dot(b, b)
    
    if dot_bb == 0:
        # Handle case where vector b is the zero vector
        p_flat = np.zeros_like(b)
    else:
        # MATLAB: p = dot(a,b)/dot(b,b) * b; 
        scalar_factor = dot_ab / dot_bb
        p_flat = scalar_factor * b
    
    # MATLAB: if (iscolumn(a)) ... else ...
    # This checks the orientation of the original input 'a' to determine 
    # the orientation of the output 'p'.

    a_orig = np.array(a).squeeze()
    
    if a_orig.ndim > 1 and a_orig.shape[0] > a_orig.shape[1]:
        # If 'a' was a column vector, use 'vertical' (N x 1)
        # Assuming 'vertical' is defined or imported, or using manual reshape
        p = p_flat.reshape(-1, 1)
    else:
        # If 'a' was a row vector (or 1D), transpose the vertical result (1 x N)
        # MATLAB: p = vertical(p)'; 
        p = p_flat.reshape(1, -1)
        
    return p