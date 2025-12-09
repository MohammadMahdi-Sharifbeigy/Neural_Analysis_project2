import numpy as np

def fdr_bh(pvals, q=0.05, method='pdep', report='no'):
    """
    MATLAB: function [h, crit_p, adj_p]=fdr_bh(pvals,q,method,report);
    
    Benjamini-Hochberg (1995) and Benjamini-Yekutieli (2001) procedure 
    for controlling the False Discovery Rate (FDR).
    
    Args:
        pvals (np.ndarray): Vector or matrix of p-values.
        q (float): The desired false discovery rate. Default=0.05.
        method (str): 'pdep' (for independent/positive dependent tests, DEFAULT) 
                      or 'dep' (for any dependency structure).
        report (str): 'yes' or 'no' (controls print output). Default='no'.
        
    Returns:
        tuple: (h, crit_p, adj_p) - 
               h (bool array): Significance mask (True=reject H0, pvals <= crit_p).
               crit_p (float): The critical p-value threshold.
               adj_p (np.ndarray): The FDR adjusted p-values (q-values).
    """
    
    # --- Checking inputs ---
    if pvals.size == 0:
        return np.array([]), 0.0, np.array([])
    if (pvals < 0).any() or (pvals > 1).any():
        raise ValueError('p-values must be between 0 and 1.')
        
    # --- Optional inputs (default simulation) ---
    if q is None: q = 0.05
    if method is None or method == '': method = 'pdep'
    if report is None or report == '': report = 'no'

    s = pvals.shape
    
    # MATLAB: [p_sorted, sort_ids]=sort(reshape(pvals,1,prod(s)));
    pvals_flat = pvals.flatten()
    sort_ids = np.argsort(pvals_flat)
    p_sorted = pvals_flat[sort_ids]
    
    # MATLAB: [dummy, unsort_ids]=sort(sort_ids);
    # Indexes to return to original order
    unsort_ids = np.argsort(sort_ids) 
    m = len(p_sorted) # number of tests
    
    i = np.arange(1, m + 1)
    
    if method.lower() == 'pdep':
        # Benjamini & Hochberg (1995)
        # MATLAB: thresh=(1:m)*q/m;
        thresh = i * q / m
        # MATLAB: wtd_p=m*p_sorted./(1:m);
        wtd_p = m * p_sorted / i
        
    elif method.lower() == 'dep':
        # Benjamini & Yekutieli (2001)
        # MATLAB: denom=m*sum(1./(1:m));
        denom = m * np.sum(1 / i)
        # MATLAB: thresh=(1:m)*q/denom;
        thresh = i * q / denom
        # MATLAB: wtd_p=denom*p_sorted./[1:m];
        wtd_p = denom * p_sorted / i
        
    else:
        raise ValueError("Argument 'method' needs to be 'pdep' or 'dep'.")

    # --- calculate adjusted p-values (adj_p) ---
    adj_p = np.zeros(m) * np.nan
    
    # MATLAB's optimized implementation for monotonicity (D.H.J. Poot)
    # [wtd_p_sorted, wtd_p_sindex] = sort( wtd_p );
    wtd_p_sindex = np.argsort(wtd_p)
    wtd_p_sorted = wtd_p[wtd_p_sindex]
    
    nextfill = 0 # 0-based index
    for k in range(m):
        # The MATLAB logic uses 1-based indexing for nextfill and k+1
        # if wtd_p_sindex(k)>=nextfill
        if wtd_p_sindex[k] >= nextfill: 
            # adj_p(nextfill:wtd_p_sindex(k)) = wtd_p_sorted(k);
            # Note: MATLAB indexing is inclusive, Python is exclusive for end slice
            adj_p[nextfill:wtd_p_sindex[k] + 1] = wtd_p_sorted[k]
            # nextfill = wtd_p_sindex(k)+1;
            nextfill = wtd_p_sindex[k] + 1
            if nextfill >= m:
                break
                
    # MATLAB: adj_p=reshape(adj_p(unsort_ids),s);
    adj_p = adj_p[unsort_ids].reshape(s)
    
    # --- calculate h و crit_p ---
    # MATLAB: rej=p_sorted<=thresh;
    rej = p_sorted <= thresh
    
    # MATLAB: max_id=find(rej,1,'last');
    max_rej_indices = np.where(rej)[0]
    
    if max_rej_indices.size == 0:
        crit_p = 0.0
        h = pvals * 0 # All zeros
    else:
        # MATLAB: crit_p=p_sorted(max_id);
        max_id = max_rej_indices[-1]
        crit_p = p_sorted[max_id]
        
        # MATLAB: h=pvals<=crit_p;
        h = pvals <= crit_p

    # --- Report ---
    if report.lower() == 'yes':
        n_sig = np.sum(h)
        if n_sig == 1:
            print(f'Out of {m} tests, {n_sig} is significant using a false discovery rate of {q}.')
        else:
            print(f'Out of {m} tests, {n_sig} are significant using a false discovery rate of {q}.')
            
        if method.lower() == 'pdep':
            print('FDR procedure used is guaranteed valid for independent or positively dependent tests.')
        else:
            print('FDR procedure used is guaranteed valid for independent or dependent tests.')
            
    return h, crit_p, adj_p