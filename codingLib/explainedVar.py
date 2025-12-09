import numpy as np

def explainedVar(a, b):
    """
    MATLAB: function y = explainedVar(a,b)
    
    Calculates the proportion of variance in 'a' that is explained by 'b' 
    (equivalent to R-squared).
    
    Args:
        a (list or np.ndarray): True data vector.
        b (list or np.ndarray): Predicted/explaining data vector.
        
    Returns:
        float: Explained variance (R-squared).
    """
    
    a = np.array(a).flatten()
    b = np.array(b).flatten()
    
    # Find indices where either a or b is NaN/Inf
    valid_mask = ~np.isnan(a) & ~np.isnan(b) & ~np.isinf(a) & ~np.isinf(b)
    
    a_valid = a[valid_mask]
    b_valid = b[valid_mask]
    
    if len(a_valid) < 2:
        return np.nan

    # MATLAB: totVar = nanmean((a - nanmean(a)).^2); 
    # Total Variance (Mean Squared Error from the Mean)
    # We use np.mean since we removed NaNs/Infs
    mean_a = np.mean(a_valid)
    totVar = np.mean((a_valid - mean_a)**2)
    
    # MATLAB: resVar = nanmean((a - b).^2);
    # Residual Variance (Mean Squared Error from the Explaining vector)
    resVar = np.mean((a_valid - b_valid)**2)
    
    if totVar == 0:
        # If total variance is zero, the data is constant. Explained variance is undefined or 1 if residual is also zero.
        return 1.0 if resVar == 0 else np.nan
        
    # MATLAB: y = 1 - resVar/totVar; 
    y = 1 - resVar / totVar
    
    return y

# Example usage (for testing):
# a_test = np.array([1, 2, 3, 4, 5])
# b_test = np.array([1.1, 2.1, 3.1, 4.1, 5.1]) # High R^2
# explainedVar(a_test, b_test)