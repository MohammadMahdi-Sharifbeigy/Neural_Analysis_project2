import numpy as np

def roc(data):
    """
    MATLAB: function [tp,fp] = roc(data)
    Receiver Operating Characteristic (ROC) curve.
    
    Args:
        data (np.ndarray): Nx2 matrix [t, y]. 
                           t: class value (>0 positive, <=0 negative), 
                           y: score value.
                           
    Returns:
        tuple: (tp, fp) - True Positive Rate and False Positive Rate.
    """
    if data.shape[1] != 2:
        raise ValueError('Incorrect input size in ROC!')
    
    t = data[:, 0]
    y = data[:, 1]
    
    t = t > 0
    
    # MATLAB: [Y,idx] = sort(-y); t = t(idx);
    # Sort by score (y) in descending order (with -y sorting ascending)
    # and apply the same order to labels (t)
    idx = np.argsort(-y)
    t = t[idx]
    
    # MATLAB: tp = cumsum(t)/sum(t); fp = cumsum(~t)/sum(~t);
    # Calculate TPR and FPR (cumulative rates)
    tp = np.cumsum(t) / np.sum(t)
    fp = np.cumsum(~t) / np.sum(~t)
    
    # MATLAB: [uY,idx] = unique(Y); tp = tp(idx); fp = fp(idx);
    # Ties management: Keeps only points that are unique in the scoring.
    # Note: Y = -y[idx].
    Y_sorted = -y[idx]
    _, unique_indices = np.unique(Y_sorted, return_index=True)
    tp = tp[unique_indices]
    fp = fp[unique_indices]
    
    # MATLAB: add trivial end-points
    # Add endpoints (0,0) and (1,1)
    # [0 ; tp ; 1]
    tp = np.concatenate(([0], tp, [1]))
    fp = np.concatenate(([0], fp, [1]))
    
    return tp, fp

# Usage
# y = np.array([1, 2, 3, 4, 5])
# t = np.array([0, 0, 1, 1, 1])
# data_test = np.column_stack((t, y))
# tp, fp = roc(data_test)
# print("TP:", tp)
# print("FP:", fp)