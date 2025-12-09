import numpy as np
from scipy.stats import pearsonr

def corrcoeff(x, y):
    """
    MATLAB: function [c,p] = corrcoeff(x,y)
    
    Calculates the Pearson correlation coefficient (c) and its p-value (p) 
    between two vectors, automatically handling NaN/Inf values.
    
    Args:
        x (list or np.ndarray): First vector.
        y (list or np.ndarray): Second vector.
        
    Returns:
        tuple: (c, p) - Correlation coefficient and p-value.
    """
    
    x = np.array(x).flatten()
    y = np.array(y).flatten()
    
    if x.shape != y.shape:
        raise ValueError("Vectors x and y must have the same length.")
        
    # MATLAB: iInvalid = find(isnan(x+y) | isinf(x+y)); 
    # Find indices where either x or y is NaN or Inf
    
    # Check for NaN or Inf in x or y
    invalid_mask = np.isnan(x) | np.isinf(x) | np.isnan(y) | np.isinf(y)
    
    # MATLAB: iValid = setdiff(1:length(x),iInvalid);
    # Select valid data points
    iValid = ~invalid_mask
    
    x_valid = x[iValid]
    y_valid = y[iValid]
    
    if len(x_valid) < 2:
        # Not enough valid data points for correlation
        c = np.nan
        p = np.nan
        return c, p
    
    # MATLAB: [R,P] = corrcoef(x(iValid), y(iValid));
    # Use Pearsonr in SciPy, which is a common way to get r and p.
    # SciPy's pearsonr is similar to the relevant part of MATLAB's corrcoef(x, y).
    
    # pearsonr returns (r, p-value)
    # MATLAB's corrcoef returns a 2x2 matrix R and P for two inputs. 
    # The actual correlation is R(1, 2) and the p-value is P(1, 2).
    
    r_val, p_val = pearsonr(x_valid, y_valid)
    
    # MATLAB: if (size(R,2)>1) ... else ...
    # This check is inherently true if we have enough valid points, as we are 
    # calculating the correlation between x and y (1 vs 1 variable).
    
    c = r_val
    p = p_val
    
    return c, p