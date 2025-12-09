import numpy as np
from scipy.ndimage import uniform_filter1d

# --- Assumed smoothBinary implementation (Placeholder) ---
def smoothBinary(y_row, win, method=''):
    """
    Assumed smoothBinary function: performs smoothing on a 1D vector.
    Placeholder uses a simple uniform (boxcar) filter, ignoring NaNs (requires nan-aware filter).
    
    Args:
        y_row (np.ndarray): 1D data row.
        win (int): Smoothing window width.
        method (str): Smoothing method (ignored here).
        
    Returns:
        np.ndarray: Smoothed row.
    """
    # This is a naive implementation; a true nan-aware smoothBinary 
    # is required for exact functional match with the MATLAB code's intent.
    
    # For a simple smooth, we use uniform_filter1d (boxcar)
    # The MATLAB smoothBinary likely handles NaN by ignoring them in the average.
    
    # To mimic a nan-aware mean filter (moving average):
    y_row = np.array(y_row, dtype=float)
    sy_row = np.full_like(y_row, np.nan)
    half_win = int(np.floor(win / 2))
    L = len(y_row)
    
    for k in range(L):
        start = max(0, k - half_win)
        end = min(L, k + half_win + 1)
        window_data = y_row[start:end]
        sy_row[k] = np.nanmean(window_data)
        
    return sy_row
# --------------------------------------------------------

def smoothD(y, dim, win):
    """
    MATLAB: function sy = smoothD(y,dim,win)
    
    Applies smoothing (using 'smoothBinary') along a specified dimension 
    of a 2D array 'y'.
    
    Args:
        y (np.ndarray): 2D array to be smoothed.
        dim (int): Dimension along which to smooth (1 for rows, 2 for columns).
        win (int): Smoothing window width.
        
    Returns:
        np.ndarray: Smoothed array 'sy'.
    """
    
    y = np.array(y)
    
    # MATLAB uses 1-based indexing for dim: 1=rows, 2=columns.
    # If dim=1, transpose to smooth along rows (the 2nd dimension in MATLAB's loops).
    if dim == 1:
        # Transpose so that the dimension to be smoothed (columns in the loop) 
        # becomes the second dimension.
        y_proc = y.T
    elif dim == 2:
        y_proc = y
    else:
        raise ValueError("Dimension 'dim' must be 1 or 2.")

    # Initialize output array with NaNs
    sy_proc = np.full_like(y_proc, np.nan, dtype=float)
    
    # Loop over the first dimension (each row/column vector to be smoothed)
    for i in range(y_proc.shape[0]):
       # sy(i,:) = smoothBinary(y(i,:),win, ''); 
       sy_proc[i, :] = smoothBinary(y_proc[i, :], win)
       
    # Transpose back if the original input was transposed
    if dim == 1:
        sy = sy_proc.T
    else:
        sy = sy_proc
        
    return sy