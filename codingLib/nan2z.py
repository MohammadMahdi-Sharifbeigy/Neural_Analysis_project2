import numpy as np

def nan2z(a):
    """
    MATLAB: function b = nan2z(a)
    
    Replaces NaN values in array 'a' with zero.
    
    Args:
        a (np.ndarray): Input array that may contain NaN values.
        
    Returns:
        np.ndarray: Output array with NaNs replaced by 0.
    """
    b = np.copy(a) # Make a copy to avoid modifying the input array
    
    # MATLAB: b(isnan(b)) = 0;
    b[np.isnan(b)] = 0
    
    return b