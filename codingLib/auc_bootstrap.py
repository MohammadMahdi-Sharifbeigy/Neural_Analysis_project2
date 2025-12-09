import numpy as np
from .auc import auc_value_only

def auc_bootstrap(data, nboot=1000, flag='both', H0=0.5):
    """
    MATLAB: function p = auc_bootstrap(data,nboot,flag,H0)
    
    Bootstrap test if AUC is different from H0 (Null Hypothesis).
    
    Args:
        data (np.ndarray): Nx2 matrix [t, y]. t: class, y: score.
        nboot (int): Number of resamples. Default=1000.
        flag (str): 'both' (two-tailed, DEFAULT), 'upper' (right-tailed), 
                    'lower' (left-tailed).
        H0 (float): Null hypothesis value (default=0.5).
        
    Returns:
        float: p-value.
    """
    
    if data.shape[1] != 2:
       raise ValueError('Incorrect input size in AUC_BOOTSTRAP!')
       
    # --- Optional inputs (default simulation) ---
    if H0 is None: H0 = 0.5
    if flag is None or flag == '': flag = 'both'
    flag = flag.lower()
    if nboot is None: nboot = 1000

    N = data.shape[0]
    A_boot = np.zeros(nboot)
    
    for i in range(nboot):
        # MATLAB: ind = unidrnd(N,[N 1]); (Uniform Discrete Random Sampling)
        # Python: np.random.randint(low, high, size)
        # MATLAB is from 1 to N (inclusive). Python is from 0 to N (exclusive).
        # We only need the indices: [0, N-1]
        ind = np.random.randint(0, N, N) 
        
        # MATLAB: A_boot(i) = auc(data(ind,:));
        try:
             A_boot[i] = auc_value_only(data[ind, :])
        except NameError:
             raise NameError("The 'auc_value_only' function (from auc.py logic) is required but not found in scope.")
        
    # --- Calculate P-Value ---
    
    # MATLAB: ltpv = mean(A_boot <= H0); (Lower-tailed test)
    ltpv = np.mean(A_boot <= H0)
    
    # MATLAB: utpv = mean(A_boot >= H0); (Upper-tailed test)
    utpv = np.mean(A_boot >= H0)
    
    # MATLAB: ttpv = 2*min(ltpv,utpv); (Two-tailed test)
    ttpv = 2 * min(ltpv, utpv)
    
    if flag == 'upper':
       # Tests AUC is not larger than H0. Rejects if AUC is significantly lower.
       p = ltpv
    elif flag == 'lower':
       # Tests AUC is not smaller than H0. Rejects if AUC is significantly higher.
       p = utpv
    else: # 'both'
       p = ttpv
       
    return p