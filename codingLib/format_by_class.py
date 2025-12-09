import numpy as np

def format_by_class(dp, dn):
    """
    MATLAB: function data = format_by_class(dp,dn)
    
    Formats two score vectors (dp and dn) into a single [class, score] matrix.
    
    Args:
        dp (list or np.ndarray): Scores for the positive ("signal") distribution.
        dn (list or np.ndarray): Scores for the negative ("noise") distribution.
        
    Returns:
        np.ndarray: A matrix with two columns: [class_label, score].
                    Class labels are 1 (positive) and 0 (negative).
    """
    
    # MATLAB: dp = dp(:); dn = dn(:); (Ensure column vector shape)
    dp = np.array(dp).flatten()
    dn = np.array(dn).flatten()
    
    # MATLAB: y = [dp ; dn]; (Concatenate scores)
    y = np.concatenate((dp, dn))
    
    # MATLAB: t = logical([ ones(size(dp)) ; zeros(size(dn)) ]); (Create class labels)
    # Use integer labels 1 and 0 for compatibility with standard classification tools
    t = np.concatenate((np.ones_like(dp, dtype=int), np.zeros_like(dn, dtype=int)))
    
    # MATLAB: data = [t,y]; (Combine into an Nx2 matrix)
    # np.column_stack is the equivalent of [t, y] in MATLAB
    data = np.column_stack((t, y))
    
    return data