import numpy as np

def z2nan(a):
    """
    MATLAB: function b = z2nan(a)
    
    Replaces zero values in array 'a' with NaN.
    
    Args:
        a (np.ndarray): Input array.
        
    Returns:
        np.ndarray: Output array with zeros replaced by NaN.
    """
    
    # Ensure the array can hold NaN (i.e., is a float type)
    if not np.issubdtype(a.dtype, np.floating):
        b = a.astype(float)
    else:
        b = np.copy(a)
        
    # MATLAB: b(b==0) = nan; 
    b[b == 0] = np.nan
    
    return b