import numpy as np

def vertical(x):
    """
    MATLAB: function y = vertical(x)
    
    Ensures the input array 'x' is a column vector (N x 1). 
    If x is already 1D, it is reshaped to (N, 1). If x is a row matrix 
    (1 x N), it is transposed. If x is already (N, M) or (N, 1), it is returned as is.
    
    Args:
        x (np.ndarray): Input array.
        
    Returns:
        np.ndarray: The input array as a column vector (N, 1) or an N x M matrix.
    """
    
    y = np.array(x)
    
    # MATLAB: if (size(x,1) < size(x,2)) -> check if it's a row vector (or 1D)
    # Check if the number of rows is less than the number of columns (excluding scalars)
    if y.ndim == 1:
        # If 1D, reshape to column vector (N, 1)
        y = y.reshape(-1, 1)
    elif y.shape[0] < y.shape[1]:
        # If row vector (1 x N), transpose it
        y = y.T
        
    return y