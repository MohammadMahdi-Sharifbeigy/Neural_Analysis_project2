import numpy as np
from scipy.interpolate import interp1d


def dip_test(x, delta_x=0):
    """
    Hartigan's dip test of unimodality
    
    Parameters
    ----------
    x : array_like
        Vector of observations
    delta_x : float, optional
        Spacing of a discrete distribution (default: 0 for continuous)
        
    Returns
    -------
    p : float
        P-value for rejecting the null hypothesis that the distribution is unimodal.
        Note: This p-value is not based on exact calculation but on a look-up table
        constructed using simulations with interpolation.
    dip : float
        The dip statistic
    xl : float
        The lower end of the modal interval
    xu : float
        The upper end of the modal interval
        
    Notes
    -----
    This is an implementation of the dip test of unimodality published in:
    J. A. Hartigan, P. M. Hartigan: The Dip Test of Unimodality. Annals of Statistics,
    13 (1985) 70-84.
    
    Based on Fortran algorithm in:
    P. M. Hartigan: Algorithm AS 217: Computation of the Dip Statistic to Test for
    Unimodality. Applied Statistics, 34 (1985) 320-325.
    
    Also incorporates correction from:
    C. J. Sommer, J. N. McNamara: Power considerations for the dip test of unimodality
    using mixtures of normal and uniform distributions. American Statistical Association:
    Proceedings of the Statistical Computing Section, 1987, 186-191.
    
    Extensions:
    1) Extended table for conservative results with larger sample sizes
    2) Extension for handling discrete distributions
    """
    
    # Check arguments
    x = np.asarray(x).flatten()
    
    if x.ndim != 1:
        raise ValueError('DIP_TEST: x must be a vector!')
    
    if len(x) < 15:
        raise ValueError('DIP_TEST: x is too short!')
    
    if delta_x < 0:
        raise ValueError('DIP_TEST: delta_x must not be negative!')
    
    # Check plausibility of given spacing
    if delta_x > 0:
        temp = np.sort(x)
        temp = np.diff(temp)
        temp = temp[temp != 0]
        temp = np.unique(temp)  # get all non-zero spacings
        temp_normalized = temp / delta_x
        
        if not np.allclose(temp_normalized, np.round(temp_normalized)):
            raise ValueError('DIP_TEST: The given delta_x is incompatible with x!')
    
    # Make continuous values from discrete values
    if delta_x > 0:
        x = x + (np.random.rand(len(x)) - 0.5) * delta_x
    
    # Sort the vector
    x = np.sort(x)
    n = len(x)
    
    # Further checks
    if x[0] == x[n-1]:
        raise ValueError('DIP_TEST: All observations must not be identical!')
    
    # Computation
    low = 0  # lower index (0-based in Python)
    high = n - 1  # upper index
    dip = 1 / n
    xl = x[low]
    xu = x[high]
    
    # Establish the indices over which combination is necessary for the convex minorant fit
    mn = np.zeros(n, dtype=int)
    mn[0] = 0
    
    for j in range(1, n):
        mn[j] = j - 1
        
        while True:
            mnj = mn[j]
            mnmnj = mn[mnj]
            a = mnj - mnmnj
            b = j - mnj
            
            if (mnj == 0) or (((x[j] - x[mnj]) * a) < ((x[mnj] - x[mnmnj]) * b)):
                break
            
            mn[j] = mnmnj
    
    # Establish the indices over which combination is necessary for the concave majorant fit
    mj = np.zeros(n, dtype=int)
    mj[n-1] = n - 1
    
    for jk in range(1, n):
        k = n - jk
        mj[k] = k + 1
        
        while True:
            mjk = mj[k]
            mjmjk = mj[mjk]
            a = mjk - mjmjk
            b = k - mjk
            
            if (mjk == n - 1) or (((x[k] - x[mjk]) * a) < ((x[mjk] - x[mjmjk]) * b)):
                break
            
            mj[k] = mjmjk
    
    # Start the cycling
    # Collect the change points for the GCM from high to low
    while True:  # line 40 in the Fortran algorithm (big loop)
        gcm = [high]
        
        while True:
            igcm1 = gcm[-1]
            gcm.append(mn[igcm1])
            
            if gcm[-1] <= low:
                break
        
        icx = len(gcm)
        
        # Collect the change points for the LCM from low to high
        lcm = [low]
        
        while True:
            lcm1 = lcm[-1]
            lcm.append(mj[lcm1])
            
            if lcm[-1] >= high:
                break
        
        icv = len(lcm)
        
        ig = icx - 1
        ih = icv - 1
        
        # Find the largest distance greater than "dip" between the GCM and the LCM from low to high
        ix = icx - 2
        iv = 1
        d = 0
        
        if (icx != 2) or (icv != 2):  # lines 50 - 60 in the Fortran algorithm
            while True:  # line 50 in the Fortran algorithm
                igcmx = gcm[ix]
                lcmiv = lcm[iv]
                
                if igcmx > lcmiv:
                    # If the next point of either the GCM or LCM is from the GCM then calculate distance here
                    lcmiv = lcm[iv]  # line 55 in the Fortran algorithm
                    igcm = gcm[ix]
                    igcm1 = gcm[ix + 1]
                    a = lcmiv - igcm1 + 1
                    b = igcm - igcm1
                    dx = a / n - ((x[lcmiv] - x[igcm1]) * b) / (n * (x[igcm] - x[igcm1]))
                    iv = iv + 1
                    
                    if dx >= d:
                        d = dx
                        ig = ix + 1
                        ih = iv - 1
                
                else:
                    # If the next point of either the GCM or LCM is from the LCM then calculate distance here
                    lcmiv1 = lcm[iv - 1]
                    a = lcmiv - lcmiv1
                    b = igcmx - lcmiv1 - 1
                    dx = ((x[igcmx] - x[lcmiv1]) * a) / (n * (x[lcmiv] - x[lcmiv1])) - b / n
                    ix = ix - 1
                    
                    if dx >= d:
                        d = dx
                        ig = ix + 1
                        ih = iv
                
                if ix < 1:  # line 60 in the Fortran algorithm
                    ix = 1
                
                if iv > icv - 1:
                    iv = icv - 1
                
                if gcm[ix] == lcm[iv]:
                    break  # leave while loop
        
        else:
            d = 1 / n
        
        # line 65 in the Fortran algorithm
        if d < dip:  # Are we done?
            dip = dip / 2
            xl = x[low]
            xu = x[high]
            break  # leave the main loop
        
        # Calculate the dips for the current low and high
        # The dip for the convex minorant
        dl = 0
        
        if ig != icx - 1:
            icxa = icx - 1
            
            for j in range(ig, icxa):
                temp = 1 / n
                jb = gcm[j + 1]
                je = gcm[j]
                
                if ((je - jb) > 1) and (x[je] != x[jb]):
                    a = je - jb
                    const = a / (n * (x[je] - x[jb]))
                    
                    for jr in range(jb, je + 1):
                        b = jr - jb + 1
                        t = b / n - (x[jr] - x[jb]) * const
                        
                        if t > temp:
                            temp = t
                
                if dl < temp:
                    dl = temp
        
        # The dip for the concave majorant
        du = 0
        
        if ih != icv - 1:
            icva = icv - 1
            
            for k in range(ih, icva):
                temp = 1 / n
                kb = lcm[k]
                ke = lcm[k + 1]
                
                if ((ke - kb) > 1) and (x[ke] != x[kb]):
                    a = ke - kb
                    const = a / (n * (x[ke] - x[kb]))
                    
                    for kr in range(kb, ke + 1):
                        b = kr - kb - 1
                        t = (x[kr] - x[kb]) * const - b / n
                        
                        if t > temp:
                            temp = t
                
                if du < temp:
                    du = temp
        
        # Determine the current maximum
        dipnew = dl
        
        if du > dl:
            dipnew = du
        
        if dip < dipnew:
            dip = dipnew
        
        low = gcm[ig]
        high = lcm[ih]
    
    # Calculate the p value (based on the table given in the publications)
    # Extended table for sample sizes of 500 and 1000
    nvec = np.array([15, 20, 30, 50, 100, 200, 500, 1000])
    dcell = [
        np.array([0.0544, 0.0606, 0.0641, 0.0836, 0.1097, 0.1179, 0.1365, 0.1424, 0.1538]),
        np.array([0.0474, 0.0529, 0.0569, 0.0735, 0.0970, 0.1047, 0.1209, 0.1262, 0.1382]),
        np.array([0.0395, 0.0442, 0.0473, 0.0617, 0.0815, 0.0884, 0.1012, 0.1061, 0.1177]),
        np.array([0.0312, 0.0352, 0.0378, 0.0489, 0.0645, 0.0702, 0.0804, 0.0842, 0.0926]),
        np.array([0.0228, 0.0256, 0.0274, 0.0355, 0.0471, 0.0510, 0.0586, 0.0619, 0.0987]),
        np.array([0.0165, 0.0185, 0.0197, 0.0255, 0.0341, 0.0370, 0.0429, 0.0449, 0.0496]),
        np.array([0.0107, 0.0119, 0.0127, 0.0164, 0.0218, 0.0237, 0.0270, 0.0287, 0.0311]),
        np.array([0.0075, 0.0085, 0.0091, 0.0117, 0.0155, 0.0168, 0.0197, 0.0206, 0.0233])
    ]
    p_vec = np.array([0.99, 0.95, 0.9, 0.5, 0.1, 0.05, 0.01, 0.005, 0.001])
    
    # Find nearest n
    n_diff = nvec - n
    n_ind = np.where(n_diff == 0)[0]
    
    if len(n_ind) == 0:  # n not tabulated?
        n_ind = np.where(n_diff > 0)[0]
        
        if len(n_ind) == 0:  # Is there no larger n value in the table?
            n_ind = np.array([len(nvec) - 1])  # choose the largest one
    
    n_ind = n_ind[0]  # Choose (1) the exact n if it is in the table;
                       #        (2) a value of n which is tabulated, larger than the real n, and has minimum distance;
                       #            by choosing a larger n the test will be on the conservative side;
                       #        (3) the largest n in the table if the real n is larger than this value.
    
    n_comp = nvec[n_ind]  # This is the n for the test
    d_test = dip * np.sqrt(n / n_comp)  # This is the dip value for the test (interpolation based on sqrt(n)*dip)
    
    # Get the p value
    d_cell_n_ind = dcell[n_ind]
    
    if d_test < np.min(d_cell_n_ind):  # out of range?
        p = p_vec[0]  # return the maximum p value in the table
        return p, dip, xl, xu
    
    if d_test > np.max(d_cell_n_ind):  # out of range?
        p = p_vec[-1]  # return the minimum p value in the table
        return p, dip, xl, xu
    
    # Interpolation
    f = interp1d(d_cell_n_ind, p_vec, kind='cubic', fill_value='extrapolate')
    p = float(f(d_test))
    
    return p, dip, xl, xu


if __name__ == "__main__":
    # Example usage
    np.random.seed(42)
    
    # Test with unimodal distribution
    x_unimodal = np.random.normal(0, 1, 100)
    p, dip, xl, xu = dip_test(x_unimodal)
    print(f"Unimodal distribution:")
    print(f"  p-value: {p:.6f}")
    print(f"  dip: {dip:.6f}")
    print(f"  modal interval: [{xl:.6f}, {xu:.6f}]")
    print()
    
    # Test with bimodal distribution
    x_bimodal = np.concatenate([np.random.normal(-2, 0.5, 50), np.random.normal(2, 0.5, 50)])
    p, dip, xl, xu = dip_test(x_bimodal)
    print(f"Bimodal distribution:")
    print(f"  p-value: {p:.6f}")
    print(f"  dip: {dip:.6f}")
    print(f"  modal interval: [{xl:.6f}, {xu:.6f}]")
    print()
    
    # Test with discrete distribution
    x_discrete = np.array([1, 1, 2, 2, 2, 3, 3, 4, 4, 5, 5, 5, 6, 7, 7, 8, 8, 9, 9, 10] * 5)
    p, dip, xl, xu = dip_test(x_discrete, delta_x=1)
    print(f"Discrete distribution:")
    print(f"  p-value: {p:.6f}")
    print(f"  dip: {dip:.6f}")
    print(f"  modal interval: [{xl:.6f}, {xu:.6f}]")