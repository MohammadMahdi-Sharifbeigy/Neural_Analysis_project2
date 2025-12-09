import numpy as np
from scipy.interpolate import interp1d

def interpolate_cbrewer(cbrew_init, interp_method, ncolors):
    """
    MATLAB: function [interp_cmap]=interpolate_cbrewer(cbrew_init, interp_method, ncolors)
    
    Interpolates a colorbrewer map to ncolors levels.
    
    Args:
        cbrew_init (np.ndarray): The initial colormap with format N*3 (RGB).
        interp_method (str): Interpolation method ('nearest', 'linear', 'spline', 'cubic').
        ncolors (int): Desired number of colors.
        
    Returns:
        np.ndarray: The interpolated colormap (interp_cmap) with shape (ncolors, 3).
    """
    
    # MATLAB: ncolors=round(ncolors);
    ncolors = int(np.round(ncolors))

    # How many data points of the colormap available
    # MATLAB: nmax=size(cbrew_init,1);
    nmax = cbrew_init.shape[0]

    # create the associated X axis (using round to get rid of decimals)
    # MATLAB: a=(ncolors-1)./(nmax-1);
    # MATLAB: X=round([0 a:a:(ncolors-1)]);
    
    if nmax == 1:
        # If only one color, just repeat it
        interp_cmap = np.tile(cbrew_init[0, :], (ncolors, 1))
        return interp_cmap.astype(int)
        
    # The X axis represents the indices of the original data points
    # MATLAB: X runs from 0 to ncolors-1, marking the position of original points
    a = (ncolors - 1) / (nmax - 1)
    
    # Generate original X points (0, a, 2*a, ..., ncolors-1)
    # This defines where the original nmax points land on a scale from 0 to ncolors-1
    X = np.round(np.arange(nmax) * a).astype(int) 
    
    # Ensure the last point is exactly ncolors-1
    if X[-1] != ncolors - 1:
        X[-1] = ncolors - 1
        
    # X2 is the new set of points to interpolate to (0, 1, 2, ..., ncolors-1)
    # MATLAB: X2=0:ncolors-1;
    X2 = np.arange(ncolors)
    
    # --- Interpolation ---
    
    # interp1 in MATLAB uses 1-based indexing for the X vector, so X is (1:nmax).
    # Since our indices for X are customized (0, a, 2a, ...), we use X as the x-coordinates 
    # of the initial colormap and X2 as the new coordinates.
    
    # interp1d requires a 'kind' that matches the method string.
    # Note: 'cubic' in MATLAB is 'cubic' in scipy.interp1d.
    
    # We interpolate each RGB channel separately
    
    # Fix for 'cubic' interpolation: requires more than 3 points
    # Fallback to 'spline' if insufficient data points
    kind = interp_method
    if kind == 'cubic' and nmax < 4:
        kind = 'spline'
    
    # MATLAB: z=interp1(X,cbrew_init(:,1),X2,interp_method);
    # NumPy/SciPy uses 0-based indexing.
    # We must ensure X has unique elements for interp1d.
    
    unique_X, unique_indices = np.unique(X, return_index=True)
    
    z = interp1d(unique_X, cbrew_init[unique_indices, 0], kind=kind, bounds_error=False, fill_value='extrapolate')(X2)
    z2 = interp1d(unique_X, cbrew_init[unique_indices, 1], kind=kind, bounds_error=False, fill_value='extrapolate')(X2)
    z3 = interp1d(unique_X, cbrew_init[unique_indices, 2], kind=kind, bounds_error=False, fill_value='extrapolate')(X2)
    
    # MATLAB: interp_cmap=round([z' z2' z3']);
    interp_cmap = np.round(np.column_stack((z, z2, z3)))
    
    # Ensure RGB values are clipped to [0, 255] and converted to integer
    interp_cmap = np.clip(interp_cmap, 0, 255).astype(int)
    
    return interp_cmap