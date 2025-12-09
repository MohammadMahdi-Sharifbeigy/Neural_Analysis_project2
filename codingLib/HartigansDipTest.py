import numpy as np

def HartigansDipTest(xpdf):
    """
    MATLAB: function [dip,xl,xu, ifault, gcm, lcm, mn, mj]=HartigansDipTest(xpdf)
    
    Calculates Hartigan's DIP statistic for unimodality, translated directly 
    from the original FORTRAN DIPTST algorithm (AS 217).
    
    Args:
        xpdf (list or np.ndarray): Vector of sample values (empirical PDF).
        
    Returns:
        tuple: (dip, xl, xu, ifault, gcm, lcm, mn, mj) - Dip statistic, 
               modal interval ends, error flag, minorant/majorant fits and indices.
    """
    
    # --- 0. Pre-checks and Initialization ---
    x = np.sort(np.array(xpdf).flatten())
    N = len(x)
    
    # Initialize output arrays (using 1-based indexing for internal logic)
    mn = np.zeros(N + 1, dtype=int)
    mj = np.zeros(N + 1, dtype=int)
    lcm = np.zeros(N + 1)
    gcm = np.zeros(N + 1)
    ifault = 0
    
    if N <= 0:
        ifault = 1
        # print('\nHartigansDipTest. InputError : ifault=1\n')
        return 0.0, np.nan, np.nan, ifault, gcm, lcm, mn, mj
    if N == 1:
        xl = x[0]
        xu = x[0]
        dip = 0.0
        ifault = 2
        # print('\nHartigansDipTest. InputError : ifault=2\n')
        return dip, xl, xu, ifault, gcm, lcm, mn, mj
    if N > 1 and N < 4:
        xl = x[0]
        xu = x[-1]
        dip = 0.0
        ifault = 4
        # print('\nHartigansDipTest. InputError : ifault=4\n')
        return dip, xl, xu, ifault, gcm, lcm, mn, mj
    
    # Check for all values identical
    if not (x[-1] > x[0]):
        xl = x[0]
        xu = x[-1]
        dip = 0.0
        ifault = 4
        return dip, xl, xu, ifault, gcm, lcm, mn, mj

    # Check if X is perfectly unimodal (omitted in original Fortran code)
    xsign = -np.sign(np.diff(np.diff(x)))
    posi = np.where(xsign > 0)[0]
    negi = np.where(xsign < 0)[0]
    
    # The condition is: either no sign changes, or the positive changes 
    # occur entirely before the negative changes.
    is_unimodal = (len(posi) == 0) or (len(negi) == 0)
    if not is_unimodal and len(posi) > 0 and len(negi) > 0:
        # Check if the last positive sign change index is less than the first negative sign change index
        is_unimodal = np.max(posi) < np.min(negi)
        
    if is_unimodal:
        xl = x[0]
        xu = x[-1]
        dip = 0.0
        ifault = 5
        return dip, xl, xu, ifault, gcm, lcm, mn, mj

    # --- 1. Main Calculation Setup ---
    fn = N
    low = 1
    high = N
    dip = 1 / fn
    xl = x[low - 1]
    xu = x[high - 1]
    
    # --- 2. Convex Minorant Fit (mn) ---
    mn[1] = 1 # 1-based index (not used in loop)
    for j in range(2, N + 1): # 1-based index
        mn[j] = j - 1
        while True:
            mnj = mn[j]
            mnmnj = mn[mnj]
            a = mnj - mnmnj
            b = j - mnj
            
            # Check for vertical slope condition (1-based index check for x)
            condition = (mnj == 1) or \
                        ( (x[j - 1] - x[mnj - 1]) * a < (x[mnj - 1] - x[mnmnj - 1]) * b )
                        
            if condition:
                break
            mn[j] = mnmnj
            
    # --- 3. Concave Majorant Fit (mj) ---
    mj[N] = N
    na = N - 1
    for jk in range(1, na + 1):
        k = N - jk # 1-based index
        mj[k] = k + 1
        while True:
            mjk = mj[k]
            mjmjk = mj[mjk]
            a = mjk - mjmjk
            b = k - mjk
            
            # Check for vertical slope condition (1-based index check for x)
            condition = (mjk == N) or \
                        ( (x[k - 1] - x[mjk - 1]) * a < (x[mjk - 1] - x[mjmjk - 1]) * b )
                        
            if condition:
                break
            mj[k] = mjmjk

    # --- 4. Main Iteration Cycle (RECYCLE) ---
    itarate_flag = True
    
    while itarate_flag:
        # CODE BREAK POINT 40
        
        # Collect GCM change points (from HIGH to LOW)
        gcm_indices = np.zeros(N + 1, dtype=int)
        ic = 0
        gcm_indices[ic] = high
        ic += 1
        gcm_indices[ic] = mn[gcm_indices[ic - 1]]
        
        while gcm_indices[ic] > low:
            ic += 1
            gcm_indices[ic] = mn[gcm_indices[ic - 1]]
        icx = ic
        
        gcm = gcm_indices[:icx + 1] # Trim GCM indices

        # Collect LCM change points (from LOW to HIGH)
        lcm_indices = np.zeros(N + 1, dtype=int)
        ic = 0
        lcm_indices[ic] = low
        ic += 1
        lcm_indices[ic] = mj[lcm_indices[ic - 1]]
        
        while lcm_indices[ic] < high:
            ic += 1
            lcm_indices[ic] = mj[lcm_indices[ic - 1]]
        icv = ic
        
        lcm = lcm_indices[:icv + 1] # Trim LCM indices

        ig = icx
        ih = icv

        # find largest distance > 'DIP' between GCM and LCM
        ix = icx - 1
        iv = 1
        d = 0.0

        if not (icx != 1 or icv != 1): # If icx=1 and icv=1
            d = 1.0 / fn
        else:
            iterate_BP50 = True
            while iterate_BP50:
                # CODE BREAK POINT 50
                igcmx = gcm[ix]
                lcmiv = lcm[iv]
                
                goto60 = False
                
                if not (igcmx > lcmiv):
                    # LCM step (Distance from GCM point to LCM line)
                    lcmiv1 = lcm[iv - 1]
                    a = lcmiv - lcmiv1
                    b = igcmx - lcmiv1 - 1
                    
                    # 1-based index to 0-based index for x
                    x_lcmiv = x[lcmiv - 1]
                    x_lcmiv1 = x[lcmiv1 - 1]
                    x_igcmx = x[igcmx - 1]
                    
                    # Handle division by zero if vertical line in LCM
                    if x_lcmiv == x_lcmiv1:
                        dx = -b / fn
                    else:
                        dx = (x_igcmx - x_lcmiv1) * a / (fn * (x_lcmiv - x_lcmiv1)) - b / fn
                    
                    ix = ix - 1
                    
                    if dx < d:
                        goto60 = True
                    else:
                        d = dx
                        ig = ix + 1
                        ih = iv
                        goto60 = True
                else:
                    # GCM step (Distance from LCM point to GCM line)
                    # CODE BREAK POINT 55
                    igcm = gcm[ix]
                    igcm1 = gcm[ix + 1]
                    a = lcmiv - igcm1 + 1
                    b = igcm - igcm1
                    
                    # 1-based index to 0-based index for x
                    x_lcmiv = x[lcmiv - 1]
                    x_igcm = x[igcm - 1]
                    x_igcm1 = x[igcm1 - 1]
                    
                    # Handle division by zero if vertical line in GCM
                    if x_igcm == x_igcm1:
                        dx = a / fn
                    else:
                        dx = a / fn - ((x_lcmiv - x_igcm1) * b) / (fn * (x_igcm - x_igcm1))
                        
                    iv = iv + 1
                    
                    if not (dx < d):
                        d = dx
                        ig = ix + 1
                        ih = iv - 1
                    goto60 = True

                if goto60:
                    # CODE BREAK POINT 60
                    if ix < 0: ix = 0 # MATLAB ix < 1 -> Python ix < 0
                    if iv > icv: iv = icv
                    iterate_BP50 = (gcm[ix] != lcm[iv])
                
        # CODE BREAK POINT 65
        itarate_flag = not (d < dip)
        
        if itarate_flag:
            # Calculate the DIPs for the current LOW and HIGH

            # DIP for the convex minorant (dl)
            dl = 0.0
            if ig != icx:
                icxa = icx - 1
                for j in range(ig, icxa + 1):
                    temp = 1.0 / fn
                    jb = gcm[j + 1]
                    je = gcm[j]
                    
                    if not (je - jb <= 1):
                        if not (x[je - 1] == x[jb - 1]):
                            a = (je - jb)
                            const = a / (fn * (x[je - 1] - x[jb - 1]))
                            for jr in range(jb, je + 1):
                                b = jr - jb + 1
                                t = b / fn - (x[jr - 1] - x[jb - 1]) * const
                                if (t > temp): temp = t
                        
                    # CODE BREAK POINT 74
                    if (dl < temp): dl = temp
            
            # DIP for the concave majorant (du)
            # CODE BREAK POINT 80
            du = 0.0
            if not (ih == icv):
                icva = icv - 1
                for k in range(ih, icva + 1):
                    temp = 1.0 / fn
                    kb = lcm[k]
                    ke = lcm[k + 1]
                    
                    if not (ke - kb <= 1):
                        if not (x[ke - 1] == x[kb - 1]):
                            a = ke - kb
                            const = a / (fn * (x[ke - 1] - x[kb - 1]))
                            for kr in range(kb, ke + 1):
                                b = kr - kb - 1
                                t = (x[kr - 1] - x[kb - 1]) * const - b / fn
                                if (t > temp): temp = t
                                
                    # CODE BREAK POINT 86
                    if (du < temp): du = temp

            # Determine the current maximum
            # CODE BREAK POINT 90
            dipnew = dl
            if (du > dl): dipnew = du
            if (dip < dipnew): dip = dipnew
            
            low = gcm[ig]
            high = lcm[ih]
        
    # CODE BREAK POINT 100
    dip = 0.5 * dip
    xl = x[low - 1]
    xu = x[high - 1]
    
    # Extract GCM/LCM values at their change points (since they were stored as indices)
    gcm_val = gcm[gcm > 0] 
    lcm_val = lcm[lcm > 0]
    
    return dip, xl, xu, ifault, gcm_val, lcm_val, mn, mj