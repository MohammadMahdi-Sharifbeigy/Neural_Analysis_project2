import numpy as np
from scipy.stats import linregress

def vertical(v):
    """Utility function to ensure a vector is a column vector (N, 1) like MATLAB."""
    v = np.array(v)
    if v.ndim == 1:
        return v.reshape(-1, 1)
    return v

def evaluateEigVal(eigVal):
    """
    MATLAB: function [dev, eig0] = evaluateEigVal(eigVal)
    
    Estimates a linear 'noise' component (eig0) based on a central window 
    of eigenvalues and calculates the absolute deviation (dev).
    
    Args:
        eigVal (list or np.ndarray): Vector of eigenvalues.
        
    Returns:
        tuple: (dev, eig0) - 
               dev (np.ndarray): Absolute deviation from the estimated null line.
               eig0 (np.ndarray): The estimated linear null line (noise component).
    """
    
    eigVal = np.array(eigVal)
    N = len(eigVal)
    
    # MATLAB: x = round(length(eigVal)/2)+(-5:5);
    # Define the window indices for the linear fit (centered around the middle)
    center_index = int(np.round(N / 2))
    # Indices in Python are 0-based, so the middle is around N/2 - 1 for even N
    
    # MATLAB's indices (1-based): [round(N/2)-4, ..., round(N/2)+6]
    start_idx = center_index - 5
    end_idx = center_index + 5
    
    # Ensure indices are within bounds [0, N-1] for Python
    x_indices = np.arange(start_idx, end_idx + 1) 
    x_indices = x_indices[(x_indices >= 1) & (x_indices <= N)] # Filter for 1-based MATLAB range
    
    # Convert to 0-based Python indices
    python_indices = x_indices - 1
    
    if len(python_indices) < 2:
        # Handle case where window is too small
        eig0 = np.full(N, np.nan)
        dev = np.full(N, np.nan)
        print("Warning: Window for linear fit is too small.")
        return dev, eig0

    # The x-coordinates for regression (1-based MATLAB indices)
    x_regress = vertical(x_indices)
    
    # The y-coordinates for regression (the eigenvalue values)
    y_regress = vertical(eigVal[python_indices])
    
    # MATLAB: c = regress(vertical(eigVal(x)), [ones(length(x),1) x']);
    # Performs linear regression: y = c1 + c2 * x
    
    # Use standard linear algebra solution or SciPy equivalent for regress
    # [ones(length(x),1) x'] is the design matrix
    X_design = np.hstack([np.ones_like(x_regress), x_regress])
    
    # Solution c = (X^T * X)^-1 * X^T * y
    try:
        c, residuals, rank, s = np.linalg.lstsq(X_design, y_regress, rcond=None)
        c = c.flatten() # c[0] is intercept, c[1] is slope
    except np.linalg.LinAlgError:
        c = [np.nan, np.nan] # Regression failed
    
    # MATLAB: eig0 = c(1)+(1:length(eigVal))*c(2); 
    # Extrapolate the linear fit to all points
    # (1:length(eigVal)) is [1, 2, ..., N] in MATLAB
    x_all = np.arange(1, N + 1)
    
    if np.isnan(c[0]):
        eig0 = np.full(N, np.nan)
    else:
        # eig0 = intercept + slope * x_all
        eig0 = c[0] + x_all * c[1] 
        
    # MATLAB: dev = abs(vertical(eigVal) - vertical(eig0));
    dev = np.abs(vertical(eigVal) - vertical(eig0))
    
    return dev.flatten(), eig0

# Example usage (for testing):
# eigVal_test = np.exp(-(np.arange(100)/20)) + 0.01 * np.random.randn(100)
# dev_out, eig0_out = evaluateEigVal(eigVal_test)
# print("Eig0 (first 5):", eig0_out[:5])
# print("Dev (first 5):", dev_out[:5])