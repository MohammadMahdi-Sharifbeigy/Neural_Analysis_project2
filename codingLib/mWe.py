import numpy as np

def mWe(y, d=None):
    """
    MATLAB: function [m,e] = mWe(y,d)
    
    Calculates the mean (m) and Standard Error of the Mean (SEM, e) 
    of an array 'y' along dimension 'd', handling NaNs.
    
    Args:
        y (np.ndarray): Input array.
        d (int, optional): Dimension along which to operate (0-based index in Python). 
                           If None, operates on the flattened array. Default=None.
                           
    Returns:
        tuple: (m, e) - Mean and SEM.
    """
    
    # Adjust dimension: MATLAB is 1-based, Python is 0-based
    axis = d - 1 if d is not None else None
    
    # MATLAB: m = nanmean(y,d); (Calculates mean ignoring NaNs)
    m = np.nanmean(y, axis=axis)
    
    # MATLAB: nanstd(y,[],d) (Calculates standard deviation ignoring NaNs)
    std_y = np.nanstd(y, axis=axis)
    
    # MATLAB: sum(~isnan(y),d) (Counts non-NaN elements along dimension d)
    # np.sum(np.isfinite(y), axis=axis) also works if y contains only NaNs/finite numbers
    n_finite = np.sum(~np.isnan(y), axis=axis)
    
    # MATLAB: e = nanstd(y,[],d)./sqrt(sum(~isnan(y),d)); (SEM formula)
    # Handle division by zero for dimensions with no finite data points
    with np.errstate(divide='ignore', invalid='ignore'):
        e = std_y / np.sqrt(n_finite)
        
    # If the input was 1D and d was None, m and e will be scalars. 
    # If the input was higher dimensional, we return the resulting arrays.
    return m, e