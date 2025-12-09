import numpy as np

def NaNsum(x):
    """MATLAB: function y = NaNsum(x)"""
    return np.sum(x[~np.isnan(x)])

def sa(Y, smoothwidth, tol):
    """
    Internal helper function for nanfastsmooth: Sliding Average (sa).
    Handles NaNs and tolerance (tol) correctly.
    """
    Y = np.array(Y, dtype=float)
    
    if smoothwidth == 1:
        return Y
        
    # Bound Tolerance (Ensuring 0 <= tol <= 1)
    tol = np.clip(tol, 0, 1)
    
    w = int(np.round(smoothwidth))
    halfw = int(np.floor(w / 2))
    L = len(Y)
    
    # Initialize arrays to store Sums and counts (s and np)
    n = Y.shape
    s = np.zeros(n, dtype=float)
    np_count = np.zeros(n, dtype=float)

    if w % 2 != 0: # Odd window length
        # Initialise Sums and counts for the first point (k=1)
        # Y[0:halfw+1] corresponds to Y(1:halfw+1) in MATLAB (inclusive end)
        window_init = Y[0:halfw + 1] 
        SumPoints = NaNsum(window_init)
        NumPoints = np.sum(~np.isnan(window_init))
        
        s[0] = SumPoints
        np_count[0] = NumPoints
        
        # Loop through producing sum and count
        for k in range(1, L): # k=2:L in MATLAB (Python index k)
            # Remove point at the start of the window
            # MATLAB: k > halfw+1 && ~isnan(Y(k-halfw-1))
            idx_remove = k - halfw - 1
            if idx_remove >= 0 and not np.isnan(Y[idx_remove]):
                SumPoints = SumPoints - Y[idx_remove]
                NumPoints = NumPoints - 1
                
            # Add point at the end of the window
            # MATLAB: k <= L-halfw && ~isnan(Y(k+halfw))
            idx_add = k + halfw
            if idx_add < L and not np.isnan(Y[idx_add]):
                SumPoints = SumPoints + Y[idx_add]
                NumPoints = NumPoints + 1
                
            s[k] = SumPoints
            np_count[k] = NumPoints
            
    else: # Even window length
        # Initialise Sums and counts for the first point (k=1)
        # This part implements the "uneven" weights (0.5 at the edges) for even windows
        
        # MATLAB: SumPoints = NaNsum(Y(1:halfw))+0.5*Y(halfw+1);
        SumPoints = NaNsum(Y[0:halfw])
        if halfw < L and not np.isnan(Y[halfw]):
            SumPoints += 0.5 * Y[halfw]
            
        # MATLAB: NumPoints = sum(~isnan(Y(1:halfw)))+0.5;
        NumPoints = np.sum(~np.isnan(Y[0:halfw]))
        if halfw < L and not np.isnan(Y[halfw]):
            NumPoints += 0.5
            
        s[0] = SumPoints
        np_count[0] = NumPoints
        
        # Loop through producing sum and count
        for k in range(1, L): # k=2:L in MATLAB (Python index k)
            # Remove two partial points from the start
            
            # MATLAB: k > halfw+1 && ~isnan(Y(k-halfw-1))
            idx_remove_outer = k - halfw - 1
            if idx_remove_outer >= 0 and not np.isnan(Y[idx_remove_outer]):
                SumPoints = SumPoints - 0.5 * Y[idx_remove_outer]
                NumPoints = NumPoints - 0.5
                
            # MATLAB: k > halfw && ~isnan(Y(k-halfw))
            idx_remove_inner = k - halfw
            if idx_remove_inner >= 0 and not np.isnan(Y[idx_remove_inner]):
                SumPoints = SumPoints - 0.5 * Y[idx_remove_inner]
                NumPoints = NumPoints - 0.5
                
            # Add two partial points to the end
            
            # MATLAB: k <= L-halfw && ~isnan(Y(k+halfw))
            idx_add_inner = k + halfw
            if idx_add_inner < L and not np.isnan(Y[idx_add_inner]):
                SumPoints = SumPoints + 0.5 * Y[idx_add_inner]
                NumPoints = NumPoints + 0.5
                
            # MATLAB: k <= L-halfw+1 && ~isnan(Y(k+halfw-1))
            idx_add_outer = k + halfw - 1
            if idx_add_outer < L and not np.isnan(Y[idx_add_outer]):
                SumPoints = SumPoints + 0.5 * Y[idx_add_outer]
                NumPoints = NumPoints + 0.5
                
            s[k] = SumPoints
            np_count[k] = NumPoints

    # Remove the amount of interpolated datapoints desired
    # MATLAB: np(np<max((w*(1-tol)),1)) = NaN;
    threshold = np.maximum((w * (1 - tol)), 1)
    np_count[np_count < threshold] = np.nan
    
    # Calculate Smoothed Signal
    # MATLAB: SmoothY=s./np;
    with np.errstate(divide='ignore', invalid='ignore'):
        SmoothY = s / np_count
        
    return SmoothY

def nanfastsmooth(Y, w, type=1, tol=0.5):
    """
    MATLAB: function SmoothY = nanfastsmooth(Y,w,type,tol)
    
    Smooths vector Y with moving average of width w, ignoring NaNs in data.
    
    Args:
        Y (np.ndarray): Input signal.
        w (int): Window width.
        type (int): Smooth type (1=rectangular, 2=triangular, 3=pseudo-Gaussian). Default=1.
        tol (float): Tolerance for NaNs (0 to 1). Default=0.5.
        
    Returns:
        np.ndarray: Smoothed signal.
    """
    
    # Input argument handling (similar to MATLAB default assignment)
    if type is None: type = 1
    if tol is None: tol = 0.5

    Y = np.array(Y).flatten()
    
    # MATLAB: switch type case 1, 2, 3
    if type == 1:
        SmoothY = sa(Y, w, tol)
    elif type == 2:
        # Triangular: 2 passes of sliding-average
        SmoothY = sa(sa(Y, w, tol), w, tol)
    elif type == 3:
        # Pseudo-Gaussian: 3 passes of sliding-average
        SmoothY = sa(sa(sa(Y, w, tol), w, tol), w, tol)
    else:
        raise ValueError("Invalid smoothing type. Choose 1, 2, or 3.")
        
    return SmoothY