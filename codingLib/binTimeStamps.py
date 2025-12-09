import numpy as np

def binTimeStamps(t, nBins, binSize, val):
    """
    MATLAB: function bx = binTimeStamps (t,nBins, binSize, val)
    
    Bins time stamps (t) into an array of size nBins, placing 'val'
    at the index corresponding to the rounded time/binSize.
    
    Input t can be a single number or a NumPy array.
    
    Args:
        t (float or np.ndarray): The time stamp(s).
        nBins (int): The number of bins in the output array.
        binSize (float): The size of each bin.
        val (float): The value to place in the corresponding bin(s).
        
    Returns:
        np.ndarray: The binned array (bx).
    """
    
    # MATLAB's `zeros(1,nBins)` translates to a 1D NumPy array of zeros.
    bx = np.zeros(nBins)
    
    # Calculate the bin index (MATLAB: round(t/binSize)). 
    # NumPy's `round` function is used.
    bin_indices = np.round(np.array(t) / binSize).astype(int)
    
    # MATLAB indexing is 1-based, NumPy is 0-based.
    # MATLAB: bx(round(t/binSize)) = val;
    # We subtract 1 from the rounded index for 0-based indexing.
    # Note: If bin_indices results in 0, 0-1 = -1, which is valid Python indexing (last element). 
    # To mimic MATLAB's behavior, indices must be >= 1. We must ensure the indices 
    # are within the valid range [1, nBins] for the original MATLAB code.
    
    # Assuming the result of round(t/binSize) is meant to be between 1 and nBins.
    # We filter for valid indices (1 to nBins) and convert to 0-based (0 to nBins-1).
    
    valid_indices = bin_indices[(bin_indices >= 1) & (bin_indices <= nBins)]
    python_indices = valid_indices - 1
    
    if len(python_indices) > 0:
        # Assign 'val' to the corresponding 0-based index/indices
        bx[python_indices] = val
        
    return bx

# Example usage (for testing):
# print(binTimeStamps(t=5.2, nBins=10, binSize=1, val=100)) # Output: [0. 0. 0. 0. 0. 100. 0. 0. 0.] (index 5)
# print(binTimeStamps(t=[1.1, 5.2, 9.9], nBins=10, binSize=1, val=100))
# # Output: [0. 100. 0. 0. 0. 100. 0. 0. 0. 100.] (indexes 1, 5, 9)