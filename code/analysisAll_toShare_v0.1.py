"""
analysisAll_toShare_v0_1.py

Complete Python conversion of MATLAB neural data analysis script.
Converted from analysisAll_toShare_v0_1.m

This script contains analysis code for neural data including:
- Sample units encoding choice, side, reward etc.
- Correlation matrix for reward predictors
- Reward prediction analysis
- Effect of reward predictors on choices
- Decorrelation results
- PSTH analysis for tslp and lrr
- Population prediction analysis
- GLM continuous encoder
- CCA analysis
- Decoding reward/choice/tunp

Author: Converted to Python
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy.io import loadmat
from scipy.stats import pearsonr, spearmanr, ttest_ind, wilcoxon, ranksum
from scipy.ndimage import uniform_filter1d
from scipy.interpolate import interp1d
from scipy.signal import convolve
import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def load_mat_file(filepath):
    """
    Load .mat file using scipy and handle nested structures.
    Works with both v7 and v7.3 mat files.
    """
    try:
        data = loadmat(filepath, squeeze_me=True, struct_as_record=False)
    except NotImplementedError:
        # For v7.3 mat files, use h5py
        import h5py
        data = {}
        with h5py.File(filepath, 'r') as f:
            for key in f.keys():
                data[key] = np.array(f[key])
    return data


def extract_cell_array(mat_data, key):
    """Extract cell array from loaded mat data."""
    if key in mat_data:
        arr = mat_data[key]
        if isinstance(arr, np.ndarray):
            return arr
        return np.array([arr])
    return None


def mWe(data, dim=0):
    """
    Calculate mean and standard error along specified dimension.
    Equivalent to MATLAB mWe function.
    
    Parameters:
    -----------
    data : ndarray
        Input data array
    dim : int
        Dimension along which to compute (0 for columns, 1 for rows)
    
    Returns:
    --------
    m : ndarray
        Mean values
    e : ndarray
        Standard error values
    """
    data = np.asarray(data)
    if data.ndim == 1:
        data = data.reshape(-1, 1) if dim == 0 else data.reshape(1, -1)
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        m = np.nanmean(data, axis=dim)
        n = np.sum(~np.isnan(data), axis=dim)
        e = np.nanstd(data, axis=dim, ddof=1) / np.sqrt(n)
    return m, e


def smooth(data, window_size):
    """
    Smooth data using moving average.
    Equivalent to MATLAB smooth function.
    """
    if window_size <= 1:
        return data
    data = np.asarray(data).flatten()
    return uniform_filter1d(data, size=window_size, mode='nearest')


def jbfill(x, upper, lower, color, edge_color=None, alpha=0.5, line_alpha=1.0, ax=None):
    """
    Create filled area plot between upper and lower bounds.
    Equivalent to MATLAB jbfill function.
    """
    if ax is None:
        ax = plt.gca()
    
    x = np.asarray(x).flatten()
    upper = np.asarray(upper).flatten()
    lower = np.asarray(lower).flatten()
    
    ax.fill_between(x, lower, upper, color=color, alpha=alpha, edgecolor='none')
    return ax


def corrcoeff(x, y):
    """
    Calculate Pearson correlation coefficient and p-value.
    Handles NaN values.
    """
    x = np.asarray(x).flatten()
    y = np.asarray(y).flatten()
    
    # Remove NaN values
    mask = ~(np.isnan(x) | np.isnan(y))
    if np.sum(mask) < 3:
        return np.nan, np.nan
    
    r, p = pearsonr(x[mask], y[mask])
    return r, p


def hist2pred(X, Y, win, color, nbins=50, jit=0, plot_flag=1, ax=None):
    """
    Create histogram-based prediction plot.
    Equivalent to MATLAB hist2pred function.
    """
    if ax is None:
        ax = plt.gca()
    
    X = np.asarray(X).flatten()
    Y = np.asarray(Y).flatten()
    
    # Remove NaN values
    mask = ~(np.isnan(X) | np.isnan(Y))
    X = X[mask]
    Y = Y[mask]
    
    # Bin X values
    percentiles = np.linspace(0, 100, nbins + 1)
    bins = np.percentile(X, percentiles)
    bins = np.unique(bins)
    
    if len(bins) < 2:
        return np.nan, np.nan
    
    bin_indices = np.digitize(X, bins) - 1
    bin_indices = np.clip(bin_indices, 0, len(bins) - 2)
    
    # Calculate mean Y for each bin
    bin_means = []
    bin_centers = []
    bin_errors = []
    
    for i in range(len(bins) - 1):
        mask = bin_indices == i
        if np.sum(mask) > 0:
            bin_means.append(np.nanmean(Y[mask]))
            bin_centers.append((bins[i] + bins[i + 1]) / 2)
            bin_errors.append(np.nanstd(Y[mask]) / np.sqrt(np.sum(mask)))
    
    bin_means = np.array(bin_means)
    bin_centers = np.array(bin_centers)
    bin_errors = np.array(bin_errors)
    
    if plot_flag and len(bin_centers) > 0:
        ax.errorbar(bin_centers + jit, bin_means, yerr=bin_errors, 
                   color=color, fmt='o-', capsize=0, markersize=4)
    
    r, p = corrcoeff(X, Y)
    return r, p


def fdr_bh(p_values, alpha=0.05):
    """
    Benjamini-Hochberg FDR correction.
    
    Parameters:
    -----------
    p_values : array-like
        Array of p-values
    alpha : float
        Desired FDR level
    
    Returns:
    --------
    h : ndarray
        Boolean array of rejected hypotheses
    crit_p : float
        Critical p-value threshold
    adj_p : ndarray
        Adjusted p-values
    """
    p_values = np.asarray(p_values).flatten()
    n = len(p_values)
    
    # Sort p-values
    sorted_idx = np.argsort(p_values)
    sorted_p = p_values[sorted_idx]
    
    # Calculate adjusted p-values
    adj_p = np.zeros(n)
    for i in range(n):
        adj_p[sorted_idx[i]] = sorted_p[i] * n / (i + 1)
    
    # Ensure monotonicity
    adj_p = np.minimum.accumulate(adj_p[::-1])[::-1]
    adj_p = np.minimum(adj_p, 1.0)
    
    # Find threshold
    threshold_line = np.arange(1, n + 1) * alpha / n
    below_threshold = sorted_p <= threshold_line
    
    if np.any(below_threshold):
        max_idx = np.max(np.where(below_threshold)[0])
        crit_p = threshold_line[max_idx]
    else:
        crit_p = 0
    
    h = p_values <= crit_p
    
    return h, crit_p, adj_p


def dPrime(dist0, dist1, method=1):
    """
    Calculate d-prime (sensitivity index) between two distributions.
    """
    dist0 = np.asarray(dist0).flatten()
    dist1 = np.asarray(dist1).flatten()
    
    dist0 = dist0[~np.isnan(dist0)]
    dist1 = dist1[~np.isnan(dist1)]
    
    if len(dist0) == 0 or len(dist1) == 0:
        return np.nan
    
    m0, m1 = np.mean(dist0), np.mean(dist1)
    s0, s1 = np.std(dist0, ddof=1), np.std(dist1, ddof=1)
    
    if method == 1:
        pooled_std = np.sqrt((s0**2 + s1**2) / 2)
    else:
        pooled_std = np.sqrt(s0**2 + s1**2)
    
    if pooled_std == 0:
        return np.nan
    
    return (m1 - m0) / pooled_std


def auc(y_true, y_score, alpha=0.05, method='boot', n_boot=2000):
    """
    Calculate Area Under Curve with confidence intervals.
    """
    from sklearn.metrics import roc_auc_score
    
    y_true = np.asarray(y_true).flatten()
    y_score = np.asarray(y_score).flatten()
    
    # Remove NaN values
    mask = ~(np.isnan(y_true) | np.isnan(y_score))
    y_true = y_true[mask]
    y_score = y_score[mask]
    
    if len(np.unique(y_true)) < 2:
        return np.nan, (np.nan, np.nan)
    
    try:
        auc_val = roc_auc_score(y_true, y_score)
    except:
        return np.nan, (np.nan, np.nan)
    
    # Bootstrap confidence intervals
    if method == 'boot':
        boot_aucs = []
        n = len(y_true)
        for _ in range(n_boot):
            idx = np.random.choice(n, n, replace=True)
            try:
                boot_auc = roc_auc_score(y_true[idx], y_score[idx])
                boot_aucs.append(boot_auc)
            except:
                pass
        
        if len(boot_aucs) > 0:
            ci = (np.percentile(boot_aucs, 100 * alpha / 2),
                  np.percentile(boot_aucs, 100 * (1 - alpha / 2)))
        else:
            ci = (np.nan, np.nan)
    else:
        ci = (np.nan, np.nan)
    
    return auc_val, ci


def format_by_class(class0, class1):
    """
    Format data for AUC calculation.
    Returns (y_true, y_score) tuple.
    """
    class0 = np.asarray(class0).flatten()
    class1 = np.asarray(class1).flatten()
    
    y_score = np.concatenate([class0, class1])
    y_true = np.concatenate([np.zeros(len(class0)), np.ones(len(class1))])
    
    return y_true, y_score


def cleanFigure(ax=None):
    """
    Clean up figure appearance.
    Equivalent to MATLAB cleanFigure function.
    """
    if ax is None:
        ax = plt.gca()
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(direction='out')
    ax.set_facecolor('none')


def z2nan(x):
    """Convert zeros to NaN."""
    x = np.asarray(x, dtype=float)
    x[x == 0] = np.nan
    return x


def jit(n, scale=0.05):
    """Generate random jitter values."""
    return np.random.uniform(-scale, scale, n) if isinstance(n, int) else np.random.uniform(-scale, scale)


def binTimeStamps(timestamps, n_bins, bin_size, values=None):
    """
    Bin timestamps into specified bins.
    
    Parameters:
    -----------
    timestamps : array-like
        Event timestamps
    n_bins : int
        Number of bins
    bin_size : float
        Size of each bin
    values : array-like, optional
        Values associated with each timestamp
    
    Returns:
    --------
    binned : ndarray
        Binned data
    """
    timestamps = np.asarray(timestamps).flatten()
    binned = np.zeros(n_bins)
    
    bin_indices = (timestamps / bin_size).astype(int)
    valid = (bin_indices >= 0) & (bin_indices < n_bins)
    
    if values is None:
        values = np.ones(len(timestamps))
    else:
        values = np.asarray(values).flatten()
    
    for i, (idx, val) in enumerate(zip(bin_indices[valid], values[valid])):
        binned[idx] += val
    
    return binned


def fconv(signal, kernel, mode=''):
    """
    Fast convolution with specified mode.
    """
    signal = np.asarray(signal).flatten()
    kernel = np.asarray(kernel).flatten()
    
    if mode == 'leading':
        # Shift kernel for leading convolution
        result = convolve(signal, kernel[::-1], mode='same')
    else:
        result = convolve(signal, kernel, mode='same')
    
    return result


def nanfastsmooth(data, window, method=1, edge_fraction=0.5):
    """
    Fast smoothing with NaN handling.
    """
    data = np.asarray(data, dtype=float).flatten()
    
    # Handle NaN values
    nan_mask = np.isnan(data)
    data_filled = np.copy(data)
    
    # Interpolate over NaN values for smoothing
    if np.any(nan_mask) and not np.all(nan_mask):
        valid_idx = np.where(~nan_mask)[0]
        nan_idx = np.where(nan_mask)[0]
        data_filled[nan_mask] = np.interp(nan_idx, valid_idx, data[~nan_mask])
    
    # Apply smoothing
    smoothed = uniform_filter1d(data_filled, size=window, mode='nearest')
    
    # Restore NaN values
    smoothed[nan_mask] = np.nan
    
    return smoothed


def orthog(Y, X):
    """
    Orthogonalize Y with respect to X.
    Returns residuals and coefficients.
    """
    Y = np.asarray(Y).flatten()
    X = np.asarray(X)
    
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    
    # Add intercept
    X_with_intercept = np.column_stack([np.ones(len(Y)), X])
    
    # Remove NaN values
    mask = ~(np.isnan(Y) | np.any(np.isnan(X_with_intercept), axis=1))
    
    if np.sum(mask) < X_with_intercept.shape[1] + 1:
        return Y, np.zeros(X_with_intercept.shape[1])
    
    # Solve least squares
    try:
        coeffs, _, _, _ = np.linalg.lstsq(X_with_intercept[mask], Y[mask], rcond=None)
        residuals = Y - X_with_intercept @ coeffs
    except:
        return Y, np.zeros(X_with_intercept.shape[1])
    
    return residuals, coeffs


def trialCut(spikes, event_times, pre_time, post_time):
    """
    Cut spike data around event times.
    
    Parameters:
    -----------
    spikes : ndarray
        Spike times or binned spikes
    event_times : array-like
        Event times to align to
    pre_time : int
        Time before event (negative for before)
    post_time : int
        Time after event
    
    Returns:
    --------
    trial_spikes : ndarray
        Trial-aligned spike data
    """
    spikes = np.asarray(spikes)
    event_times = np.asarray(event_times).flatten().astype(int)
    
    window_size = post_time - pre_time + 1
    n_trials = len(event_times)
    
    if spikes.ndim == 1:
        trial_spikes = np.zeros((n_trials, window_size))
        for i, t in enumerate(event_times):
            start = t + pre_time
            end = t + post_time + 1
            if start >= 0 and end <= len(spikes):
                trial_spikes[i, :] = spikes[start:end]
    else:
        trial_spikes = np.zeros((n_trials, window_size))
        for i, t in enumerate(event_times):
            start = t + pre_time
            end = t + post_time + 1
            if start >= 0 and end <= spikes.shape[1]:
                trial_spikes[i, :] = spikes[0, start:end]
    
    return trial_spikes


def smoothD(data, method, window):
    """
    Smooth data with specified method and window.
    """
    data = np.asarray(data)
    
    if data.ndim == 1:
        return uniform_filter1d(data.astype(float), size=window, mode='nearest')
    
    # For 2D data, smooth along axis 1
    smoothed = np.zeros_like(data, dtype=float)
    for i in range(data.shape[0]):
        smoothed[i, :] = uniform_filter1d(data[i, :].astype(float), size=window, mode='nearest')
    
    return smoothed


def cbrewer(cmap_type, cmap_name, n_colors):
    """
    Create colormap similar to MATLAB cbrewer.
    """
    if cmap_type == 'div':
        if cmap_name == 'RdBu':
            colors = plt.cm.RdBu(np.linspace(0, 1, n_colors))
        elif cmap_name == 'PuOr':
            colors = plt.cm.PuOr(np.linspace(0, 1, n_colors))
        else:
            colors = plt.cm.coolwarm(np.linspace(0, 1, n_colors))
    else:
        colors = plt.cm.viridis(np.linspace(0, 1, n_colors))
    
    return colors


def chunkwiseCV(X, Y, chunks, method, extras=None):
    """
    Chunk-wise cross-validation for various prediction methods.
    
    Parameters:
    -----------
    X : ndarray
        Feature matrix
    Y : ndarray
        Target variable
    chunks : array-like
        Chunk boundaries for cross-validation
    method : str
        Method to use ('BLR', 'GLM', 'SVMR', 'CCA')
    extras : dict, optional
        Additional parameters
    
    Returns:
    --------
    result : dict
        Dictionary containing cross-validation results
    """
    from sklearn.linear_model import LogisticRegression, Ridge, LinearRegression
    from sklearn.svm import SVR
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import StandardScaler
    
    X = np.asarray(X)
    Y = np.asarray(Y).flatten()
    chunks = np.asarray(chunks).flatten()
    
    if extras is None:
        extras = {}
    
    # Handle NaN values
    nan_mask = np.isnan(Y)
    if X.ndim == 2:
        nan_mask = nan_mask | np.any(np.isnan(X), axis=1)
    
    result = {
        'Y': [],
        'predY': [],
        'aucTest': [],
        'rTest': [],
        'gofTest': []
    }
    
    # Get chunk indices
    if len(chunks) < 2:
        chunks = np.array([0, len(Y)])
    
    n_chunks = len(chunks) - 1
    
    for ch in range(n_chunks):
        start_idx = int(chunks[ch])
        end_idx = int(chunks[ch + 1])
        
        if end_idx <= start_idx:
            continue
        
        # Test indices for this chunk
        test_idx = np.arange(start_idx, end_idx)
        train_idx = np.concatenate([np.arange(0, start_idx), np.arange(end_idx, len(Y))])
        
        # Remove NaN indices
        test_idx = test_idx[~nan_mask[test_idx]]
        train_idx = train_idx[~nan_mask[train_idx]]
        
        if len(test_idx) < 2 or len(train_idx) < 2:
            result['aucTest'].append(np.nan)
            result['rTest'].append(np.nan)
            result['gofTest'].append(np.nan)
            continue
        
        X_train, X_test = X[train_idx], X[test_idx]
        Y_train, Y_test = Y[train_idx], Y[test_idx]
        
        # Standardize features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train if X_train.ndim == 2 else X_train.reshape(-1, 1))
        X_test_scaled = scaler.transform(X_test if X_test.ndim == 2 else X_test.reshape(-1, 1))
        
        try:
            if method == 'BLR':
                # Binary Logistic Regression
                model = LogisticRegression(max_iter=1000, solver='lbfgs')
                model.fit(X_train_scaled, Y_train.astype(int))
                Y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
                
                auc_val = roc_auc_score(Y_test, Y_pred_proba) if len(np.unique(Y_test)) > 1 else np.nan
                result['aucTest'].append(auc_val)
                result['rTest'].append(np.nan)
                result['gofTest'].append(auc_val)
                
            elif method == 'GLM':
                # Generalized Linear Model (using Ridge regression)
                model = Ridge(alpha=1.0)
                model.fit(X_train_scaled, Y_train)
                Y_pred = model.predict(X_test_scaled)
                
                r, _ = corrcoeff(Y_test, Y_pred)
                result['aucTest'].append(np.nan)
                result['rTest'].append(r)
                result['gofTest'].append(r)
                
            elif method == 'SVMR':
                # Support Vector Machine Regression
                kernel_scale = extras.get('kernelScale', 'auto')
                model = SVR(kernel='rbf', gamma=kernel_scale if kernel_scale != 'auto' else 'scale')
                model.fit(X_train_scaled, Y_train)
                Y_pred = model.predict(X_test_scaled)
                
                r, _ = corrcoeff(Y_test, Y_pred)
                result['aucTest'].append(np.nan)
                result['rTest'].append(r)
                result['gofTest'].append(r)
                
            elif method == 'CCA':
                # Canonical Correlation Analysis
                from sklearn.cross_decomposition import CCA
                n_comp = extras.get('nComp', 5)
                
                cca = CCA(n_components=min(n_comp, X_train_scaled.shape[1], Y_train.shape[0]))
                cca.fit(X_train_scaled, Y_train.reshape(-1, 1) if Y_train.ndim == 1 else Y_train)
                
                X_c, Y_c = cca.transform(X_test_scaled, Y_test.reshape(-1, 1) if Y_test.ndim == 1 else Y_test)
                r, _ = corrcoeff(X_c[:, 0], Y_c[:, 0] if Y_c.ndim > 1 else Y_c)
                
                result['aucTest'].append(np.nan)
                result['rTest'].append(r)
                result['gofTest'].append(r)
            
            else:
                # Default: Linear regression
                model = LinearRegression()
                model.fit(X_train_scaled, Y_train)
                Y_pred = model.predict(X_test_scaled)
                
                r, _ = corrcoeff(Y_test, Y_pred)
                result['aucTest'].append(np.nan)
                result['rTest'].append(r)
                result['gofTest'].append(r)
        
        except Exception as e:
            result['aucTest'].append(np.nan)
            result['rTest'].append(np.nan)
            result['gofTest'].append(np.nan)
        
        result['Y'].append(Y_test)
        result['predY'].append(Y_pred if 'Y_pred' in dir() else None)
    
    # Convert lists to arrays
    result['aucTest'] = np.array(result['aucTest'])
    result['rTest'] = np.array(result['rTest'])
    result['gofTest'] = np.array(result['gofTest'])
    
    return result


def constructX(var_list, filename, normalize=1):
    """
    Construct design matrix from variable list.
    
    Parameters:
    -----------
    var_list : list
        List of variable names to include
    filename : str
        Path to data file
    normalize : int
        Whether to normalize variables
    
    Returns:
    --------
    X : ndarray
        Design matrix
    cuts : dict
        Variable boundaries in X
    var_names : list
        List of variable names
    """
    data = load_mat_file(filename)
    
    X_parts = []
    cuts = {}
    var_names = []
    current_idx = 0
    
    for var in var_list:
        if var == 'respPre':
            if 'fbResp' in data:
                arr = data['fbResp'][:, :7]  # First half
                X_parts.append(arr)
                cuts['respPre'] = (current_idx, current_idx + arr.shape[1])
                current_idx += arr.shape[1]
                var_names.append('respPre')
        
        elif var == 'respPost':
            if 'fbResp' in data:
                arr = data['fbResp'][:, 7:]  # Second half
                X_parts.append(arr)
                cuts['respPost'] = (current_idx, current_idx + arr.shape[1])
                current_idx += arr.shape[1]
                var_names.append('respPost')
        
        elif var == 'rew':
            if 'fbRew' in data:
                arr = data['fbRew']
                X_parts.append(arr)
                cuts['rew'] = (current_idx, current_idx + arr.shape[1])
                current_idx += arr.shape[1]
                var_names.append('rew')
        
        elif var == 'choicePost':
            if 'fbChoice' in data:
                arr = data['fbChoice']
                X_parts.append(arr)
                cuts['choicePost'] = (current_idx, current_idx + arr.shape[1])
                current_idx += arr.shape[1]
                var_names.append('choicePost')
        
        elif var == 'tslp':
            if 'bTslp' in data:
                arr = data['bTslp'].reshape(-1, 1)
                if normalize:
                    arr = (arr - np.nanmean(arr)) / np.nanstd(arr)
                X_parts.append(arr)
                cuts['tslp'] = (current_idx, current_idx + 1)
                current_idx += 1
                var_names.append('tslp')
        
        elif var == 'lrr':
            if 'bRewRatio' in data:
                arr = data['bRewRatio'].reshape(-1, 1)
                if normalize:
                    arr = (arr - np.nanmean(arr)) / np.nanstd(arr)
                X_parts.append(arr)
                cuts['lrr'] = (current_idx, current_idx + 1)
                current_idx += 1
                var_names.append('lrr')
        
        elif var == 'loc':
            if 'bLocXs' in data and 'bLocYs' in data:
                arr = np.column_stack([data['bLocXs'], data['bLocYs']])
                X_parts.append(arr)
                cuts['loc'] = (current_idx, current_idx + 2)
                current_idx += 2
                var_names.append('loc')
    
    if X_parts:
        X = np.hstack(X_parts)
    else:
        X = np.array([])
    
    return X, cuts, var_names


def plotWeightMatrix(A, cuts, var_names, ax=None):
    """
    Plot weight matrix from CCA or other decomposition.
    """
    if ax is None:
        ax = plt.gca()
    
    im = ax.imshow(A.T, aspect='auto', cmap='RdBu_r', vmin=-1, vmax=1)
    ax.set_xlabel('Variables')
    ax.set_ylabel('Components')
    
    # Add variable separators
    for var in cuts.values():
        ax.axvline(var[0] - 0.5, color='k', linewidth=0.5)
    
    return im


def findingComp(X, Y, A, B, var_names, sel_vars, cuts, threshold=0.8):
    """
    Find components most related to selected variables.
    """
    n_comp = A.shape[1]
    n_vars = len(sel_vars)
    
    i_comp_var = np.zeros(n_vars, dtype=int)
    sg_var = np.zeros(n_vars)
    
    behav_comp_var = X @ A
    neur_comp_var = Y @ B
    
    for i, var in enumerate(sel_vars):
        if var in cuts:
            start, end = cuts[var]
            var_weights = np.abs(A[start:end, :]).mean(axis=0)
            i_comp_var[i] = np.argmax(var_weights)
            sg_var[i] = var_weights[i_comp_var[i]]
    
    return i_comp_var, sg_var, behav_comp_var, neur_comp_var


# =============================================================================
# COLOR DEFINITIONS
# =============================================================================

# Define colors used throughout the analysis
tslpCl = np.array([0, 0.6, 0])  # Green for time since last press
lrrCl = np.array([0, 0, 1])     # Blue for local reward rate
gray = np.array([0.5, 0.5, 0.5])


# =============================================================================
# MAIN ANALYSIS CLASS
# =============================================================================

class NeuralAnalysis:
    """
    Main class for neural data analysis.
    Contains all analysis functions converted from MATLAB.
    """
    
    def __init__(self, proj_path=''):
        """
        Initialize the analysis with project path.
        
        Parameters:
        -----------
        proj_path : str
            Path to the project data directory
        """
        self.proj_path = proj_path
        self.session_names = []
        self.valid_sessions = []
        self.gin_sessions = []
        self.tony_sessions = []
        self.gin_cut_off = 0
        
        # Data containers
        self.rew = {}
        self.choice = {}
        self.resp = {}
        self.n_trials = {}
        self.tslp = {}
        self.tunp = {}
        self.rew_ratio = {}
        self.i_forag = {}
        self.i_chunk = {}
        self.b_pushed_times = {}
        
    def load_session_data(self, session_idx, data_type='binnedFr200'):
        """
        Load data for a specific session.
        
        Parameters:
        -----------
        session_idx : int
            Session index
        data_type : str
            Type of data to load
        
        Returns:
        --------
        data : dict
            Loaded session data
        """
        filename = f"{self.proj_path}data/binned/{self.session_names[session_idx]}_{data_type}.mat"
        return load_mat_file(filename)
    
    # =========================================================================
    # SECTION: Sample units encoding (Fig. Example)
    # =========================================================================
    
    def plot_sample_units(self, session_idx=74, units=None, bin_size=200):
        """
        Plot sample units encoding choice, side, reward etc.
        Corresponds to MATLAB section: (EXAMPLE PLOT) Sample units encoding
        
        Parameters:
        -----------
        session_idx : int
            Session index to analyze
        units : list
            List of unit indices to plot
        bin_size : int
            Bin size in ms
        """
        if units is None:
            units = [15, 17, 19, 34, 37]
        
        s = session_idx
        
        # Load binned firing rate data
        data = self.load_session_data(s)
        btFr = data.get('btFr', None)
        trialTime = data.get('trialTime', None)
        
        if btFr is None:
            print(f"No btFr data found for session {s}")
            return
        
        # Get trial indices
        rew_s = self.rew.get(s, np.array([]))
        choice_s = self.choice.get(s, np.array([]))
        resp_s = self.resp.get(s, np.array([]))
        n_trials_s = self.n_trials.get(s, btFr.shape[1])
        
        indR = np.where(rew_s > 0)[0]
        indNR = np.where(rew_s == 0)[0]
        indSw = np.where(choice_s > 0)[0]
        indNSw = np.where(choice_s == 0)[0]
        ind1 = np.where(resp_s == 1)[0]
        ind2 = np.where(resp_s == 2)[0]
        indFirst = np.where(choice_s > 0)[0] + 1
        indCons = np.setdiff1d(np.arange(n_trials_s), indFirst)
        
        x = trialTime[::bin_size] / 1000 if trialTime is not None else np.arange(btFr.shape[2]) * bin_size / 1000
        
        smWin = 1
        
        for u in units:
            if u >= btFr.shape[0]:
                continue
                
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            
            # Subplot 1: Rewarded vs Unrewarded
            ax = axes[0, 0]
            if len(indR) > 0 and len(indNR) > 0:
                m1, e1 = mWe(np.squeeze(btFr[u, indR, :]), 0)
                m2, e2 = mWe(np.squeeze(btFr[u, indNR, :]), 0)
                
                jbfill(x, smooth(m1 + e1, smWin), smooth(m1 - e1, smWin), [1, 0, 0], alpha=0.4, ax=ax)
                jbfill(x, smooth(m2 + e2, smWin), smooth(m2 - e2, smWin), [0.4, 0.4, 0.4], alpha=0.4, ax=ax)
                ax.plot(x, smooth(m1, smWin), color=[1, 0, 0], label='Rewarded')
                ax.plot(x, smooth(m2, smWin), color=[0.4, 0.4, 0.4], label='Unrewarded')
            
            ax.set_title(f's={s}, u={u}')
            ax.set_xlabel('Time after response (s)')
            ax.set_ylabel('Firing rate (sp/s)')
            ax.legend()
            cleanFigure(ax)
            
            # Subplot 2: Switch vs Stay
            ax = axes[0, 1]
            if len(indSw) > 0 and len(indNSw) > 0:
                m1, e1 = mWe(np.squeeze(btFr[u, indSw, :]), 0)
                m2, e2 = mWe(np.squeeze(btFr[u, indNSw, :]), 0)
                
                jbfill(x, smooth(m1 + e1, smWin), smooth(m1 - e1, smWin), [1, 0.4, 0], alpha=0.4, ax=ax)
                jbfill(x, smooth(m2 + e2, smWin), smooth(m2 - e2, smWin), [0.4, 0.4, 0.4], alpha=0.4, ax=ax)
                ax.plot(x, smooth(m1, smWin), color=[1, 0.4, 0], label='Switch')
                ax.plot(x, smooth(m2, smWin), color=[0.4, 0.4, 0.4], label='Stay')
            
            ax.set_title(f's={s}, u={u}')
            ax.set_xlabel('Time after response (s)')
            ax.set_ylabel('Firing rate (sp/s)')
            ax.legend()
            cleanFigure(ax)
            
            # Subplot 3: Side 1 vs Side 2
            ax = axes[1, 0]
            if len(ind1) > 0 and len(ind2) > 0:
                m1, e1 = mWe(np.squeeze(btFr[u, ind1, :]), 0)
                m2, e2 = mWe(np.squeeze(btFr[u, ind2, :]), 0)
                
                jbfill(x, smooth(m1 + e1, smWin), smooth(m1 - e1, smWin), [0, 0, 1], alpha=0.4, ax=ax)
                jbfill(x, smooth(m2 + e2, smWin), smooth(m2 - e2, smWin), [1, 0, 0], alpha=0.4, ax=ax)
                ax.plot(x, smooth(m1, smWin), color=[0, 0, 1], label='Side 1')
                ax.plot(x, smooth(m2, smWin), color=[1, 0, 0], label='Side 2')
            
            ax.set_title(f's={s}, u={u}')
            ax.set_xlabel('Time after response (s)')
            ax.set_ylabel('Firing rate (sp/s)')
            ax.legend()
            cleanFigure(ax)
            
            # Subplot 4: First response vs Consecutive
            ax = axes[1, 1]
            if len(indFirst) > 0 and len(indCons) > 0:
                indFirst_valid = indFirst[indFirst < btFr.shape[1]]
                indCons_valid = indCons[indCons < btFr.shape[1]]
                
                if len(indFirst_valid) > 0 and len(indCons_valid) > 0:
                    m1, e1 = mWe(np.squeeze(btFr[u, indFirst_valid, :]), 0)
                    m2, e2 = mWe(np.squeeze(btFr[u, indCons_valid, :]), 0)
                    
                    jbfill(x, smooth(m1 + e1, smWin), smooth(m1 - e1, smWin), [0, 0.4, 0], alpha=0.4, ax=ax)
                    jbfill(x, smooth(m2 + e2, smWin), smooth(m2 - e2, smWin), [0.4, 0.4, 0.4], alpha=0.4, ax=ax)
                    ax.plot(x, smooth(m1, smWin), color=[0, 0.4, 0], label='First resp')
                    ax.plot(x, smooth(m2, smWin), color=[0.4, 0.4, 0.4], label='Cons resp')
            
            ax.set_title(f's={s}, u={u}')
            ax.set_xlabel('Time after response (s)')
            ax.set_ylabel('Firing rate (sp/s)')
            ax.legend()
            cleanFigure(ax)
            
            plt.tight_layout()
            plt.show()
    
    # =========================================================================
    # SECTION: Correlation matrix for reward predictors (Fig. 2b; S2d; S3a, d)
    # =========================================================================
    
    def plot_reward_predictor_correlation(self, rew_pred_all, rew_pred_name_all, 
                                          i_rew_preds=None, side_all=None, sessions_all=None):
        """
        Plot correlation matrix for selected reward predictors.
        Corresponds to MATLAB section: correlation matrix for reward predictors
        
        Parameters:
        -----------
        rew_pred_all : ndarray
            All reward predictors (n_trials x n_predictors)
        rew_pred_name_all : list
            Names of all predictors
        i_rew_preds : list
            Indices of predictors to use
        side_all : ndarray, optional
            Side information for trial selection
        sessions_all : ndarray, optional
            Session information for trial selection
        """
        if i_rew_preds is None:
            i_rew_preds = [7, 6, 4, 1]  # Default for fig 2b
        
        n_rew_preds = len(i_rew_preds)
        rew_preds = rew_pred_all[:, i_rew_preds]
        rew_pred_names = [rew_pred_name_all[i] for i in i_rew_preds]
        
        # Select trials
        if side_all is not None:
            sel_trials = np.arange(len(side_all))
        else:
            sel_trials = np.arange(rew_preds.shape[0])
        
        # Calculate correlation matrix
        R = np.zeros((n_rew_preds, n_rew_preds))
        P = np.zeros((n_rew_preds, n_rew_preds))
        
        for i in range(n_rew_preds):
            for j in range(n_rew_preds):
                R[i, j], P[i, j] = corrcoeff(rew_preds[sel_trials, i], 
                                              rew_preds[sel_trials, j])
        
        # Plot correlation matrix
        fig, ax = plt.subplots(figsize=(8, 8))
        
        cmap = plt.cm.RdBu_r
        im = ax.imshow(R, cmap=cmap, vmin=-1, vmax=1)
        
        ax.set_xticks(range(n_rew_preds))
        ax.set_xticklabels(rew_pred_names, rotation=90)
        ax.set_yticks(range(n_rew_preds))
        ax.set_yticklabels(rew_pred_names)
        
        plt.colorbar(im, ax=ax)
        ax.set_aspect('equal')
        cleanFigure(ax)
        
        plt.tight_layout()
        plt.show()
        
        return R, P
    
    # =========================================================================
    # SECTION: How does each predictor predict reward (Fig. 2a; S3b; S3e)
    # =========================================================================
    
    def plot_reward_prediction(self, rew_preds, rew_all, rew_pred_names, 
                               sessions_all=None, win=50):
        """
        Plot how each predictor predicts reward.
        Corresponds to MATLAB section: how does each predictor predicts reward
        
        Parameters:
        -----------
        rew_preds : ndarray
            Reward predictors (n_trials x n_predictors)
        rew_all : ndarray
            Reward outcomes
        rew_pred_names : list
            Names of predictors
        sessions_all : ndarray, optional
            Session information
        win : int
            Window size for histogram
        """
        n_rew_preds = rew_preds.shape[1]
        
        # Select trials (all by default)
        sel_trials = np.arange(len(rew_all))
        
        # Colors
        cl = np.array([[0, 0, 0], [0, 0, 0], lrrCl, tslpCl])
        if n_rew_preds > 4:
            cl = np.vstack([cl, np.random.rand(n_rew_preds - 4, 3)])
        
        jits = np.zeros(n_rew_preds)
        
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        axes = axes.flatten()
        
        r = np.zeros(n_rew_preds)
        p = np.zeros(n_rew_preds)
        
        for i in range(n_rew_preds):
            ax = axes[i] if i < len(axes) else plt.subplot(2, 4, i + 1)
            
            X = rew_preds[sel_trials, i]
            Y = rew_all[sel_trials]
            
            r[i], p[i] = hist2pred(X, Y, win, cl[i] if i < len(cl) else 'k', 
                                   win, jits[i], 1, ax)
            
            ax.set_xlabel(rew_pred_names[i])
            ax.set_ylim([0, 1])
            ax.set_yticks(np.arange(0, 1.1, 0.2))
            cleanFigure(ax)
        
        axes[0].set_ylabel('Reward fraction')
        
        # FDR correction
        h, c, q_val = fdr_bh(p, 0.01)
        print(f"FDR-corrected q-values: {q_val}")
        
        plt.tight_layout()
        plt.show()
        
        return r, p, q_val
    
    # =========================================================================
    # SECTION: Effect of reward predictors on choices (Fig. S2e)
    # =========================================================================
    
    def plot_predictor_effects(self, rew_pred_all, tunp_all, choice_all, rew_all, n_classes=4):
        """
        Plot the effect of reward predictors on choices and waiting time.
        Corresponds to MATLAB section: effect of reward predictors on choices
        
        Parameters:
        -----------
        rew_pred_all : ndarray
            All reward predictors
        tunp_all : ndarray
            Time until next press
        choice_all : ndarray
            Choice data (switch/stay)
        rew_all : ndarray
            Reward outcomes
        n_classes : int
            Number of classes for binning
        """
        # Log transform tunp
        tunp_all_log = np.log(tunp_all)
        tunp_all_log[choice_all == 1] = np.nan
        
        n_preds = min(5, rew_pred_all.shape[1])
        
        # Initialize storage
        d_prime_tunp = np.zeros((n_classes - 1, n_preds))
        m_switch0 = np.zeros((n_classes - 1, n_preds))
        e_switch0 = np.zeros((n_classes - 1, n_preds))
        m_tunp0 = np.zeros((n_classes - 1, n_preds))
        e_tunp0 = np.zeros((n_classes - 1, n_preds))
        m_tunp1 = np.zeros((n_classes - 1, n_preds))
        e_tunp1 = np.zeros((n_classes - 1, n_preds))
        p_tunp = np.zeros((n_classes - 1, n_preds))
        
        for j in range(n_preds):
            tmp = rew_pred_all[:, j]
            X_bins = np.percentile(tmp[~np.isnan(tmp)], np.linspace(0, 100, n_classes))
            b_X = np.digitize(tmp, X_bins) - 1
            b_X = np.clip(b_X, 0, n_classes - 2)
            n_bins = n_classes - 1
            
            for i in range(n_bins):
                ind0 = np.where((b_X == i) & (rew_all == 0))[0]
                ind1 = np.where((b_X == i) & (rew_all == 1))[0]
                
                d_prime_tunp[i, j] = dPrime(tunp_all_log[ind0], tunp_all_log[ind1], 2)
                
                tunp0 = np.exp(tunp_all_log[ind0])
                tunp1 = np.exp(tunp_all_log[ind1])
                m_tunp0[i, j], e_tunp0[i, j] = mWe(tunp0, 0)
                m_tunp1[i, j], e_tunp1[i, j] = mWe(tunp1, 0)
                
                _, p_tunp[i, j] = ttest_ind(tunp_all_log[ind0][~np.isnan(tunp_all_log[ind0])],
                                            tunp_all_log[ind1][~np.isnan(tunp_all_log[ind1])])
                
                choices0 = choice_all[ind0]
                m_switch0[i, j], e_switch0[i, j] = mWe(choices0 == 1, 0)
        
        # Colors
        cl = np.array([tslpCl, [1, 0, 0], [1, 0.7, 0], lrrCl, [1, 0, 1]])
        m_bin = np.linspace(0, 100, n_classes)
        m_bin = m_bin[:-1] + np.diff(m_bin) / 2
        
        # Plot
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Switch probability
        ax = axes[0]
        for j in range(n_preds):
            ax.errorbar(np.arange(1, n_bins + 1) + jit(1), 
                       m_switch0[:, j] * 100, 
                       yerr=e_switch0[:, j] * 100,
                       fmt='o-', color=cl[j], capsize=0, label=f'Pred {j+1}')
        
        ax.set_xlim([0, n_bins + 1])
        ax.set_xlabel('Bins (Percentile)')
        ax.set_xticks(range(1, n_bins + 1))
        ax.set_xticklabels([f'{m:.0f}' for m in m_bin])
        ax.set_ylabel('Switches when unrewarded (%)')
        ax.legend()
        cleanFigure(ax)
        
        # Waiting time difference
        ax = axes[1]
        for j in range(n_preds):
            ax.errorbar(np.arange(1, n_bins + 1) + jit(1),
                       m_tunp0[:, j] - m_tunp1[:, j],
                       yerr=e_tunp0[:, j] + e_tunp1[:, j],
                       fmt='o-', color=cl[j], capsize=0, label=f'Pred {j+1}')
        
        ax.set_xlim([0, n_bins + 1])
        ax.set_xlabel('Bins (Percentile)')
        ax.set_xticks(range(1, n_bins + 1))
        ax.set_xticklabels([f'{m:.0f}' for m in m_bin])
        ax.set_ylabel('Longer waiting time when unrewarded (s)')
        ax.legend()
        cleanFigure(ax)
        
        plt.tight_layout()
        plt.show()
        
        return d_prime_tunp, m_switch0, p_tunp
    
    # =========================================================================
    # SECTION: Effect of reward/wait on next wait (Fig. 2c, d, S3c, f)
    # =========================================================================
    
    def plot_reward_wait_effect(self, tslp_all, tunp_all, choice_all, rew_all, 
                                sessions_all=None, n_classes=4):
        """
        Plot how reward/no reward after wait affects next wait/choice.
        Corresponds to MATLAB section: reward/no rew after wait affects next wait
        
        Parameters:
        -----------
        tslp_all : ndarray
            Time since last press
        tunp_all : ndarray
            Time until next press
        choice_all : ndarray
            Choice data
        rew_all : ndarray
            Reward outcomes
        sessions_all : ndarray, optional
            Session information
        n_classes : int
            Number of classes for binning
        """
        # Select trials
        i_mon = np.arange(len(tslp_all))
        
        tmp = tslp_all[i_mon]
        var_bins = np.percentile(tmp[~np.isnan(tmp)], np.linspace(0, 100, n_classes))
        b_var_all = np.digitize(tmp, var_bins) - 1
        b_var_all = np.clip(b_var_all, 0, n_classes - 2)
        
        tunp_all_log = np.log(tunp_all[i_mon])
        tunp_all_log[choice_all[i_mon] == 1] = np.nan
        choice_all_sel = choice_all[i_mon]
        
        xs = [3, 5, 10, 30, 100]
        bins = np.arange(0.8, 5.0, 0.2)
        n_bins = n_classes - 1
        
        # Storage
        d_tunp = np.zeros(n_bins)
        p_tunp = np.zeros(n_bins)
        p_switch0 = np.zeros(n_bins)
        p_switch1 = np.zeros(n_bins)
        tunp0 = []
        tunp1 = []
        choices0 = []
        choices1 = []
        ticks = []
        
        fig, axes = plt.subplots(1, n_bins + 2, figsize=(4 * (n_bins + 2), 4))
        
        for i in range(n_bins):
            ax = axes[i]
            
            ind0 = np.where((b_var_all == i) & (rew_all[i_mon] == 0))[0]
            ind1 = np.where((b_var_all == i) & (rew_all[i_mon] == 1))[0]
            
            n0, _ = np.histogram(tunp_all_log[ind0], bins=bins)
            n1, _ = np.histogram(tunp_all_log[ind1], bins=bins)
            
            bin_centers = (bins[:-1] + bins[1:]) / 2
            
            if np.sum(n0) > 0:
                ax.plot(bin_centers, n0 / np.sum(n0), 'k', label='No reward')
            if np.sum(n1) > 0:
                ax.plot(bin_centers, n1 / np.sum(n1), 'r', label='Reward')
            
            ax.set_ylim([0, 0.25])
            
            # Plot medians
            med0 = np.nanmedian(tunp_all_log[ind0])
            med1 = np.nanmedian(tunp_all_log[ind1])
            yl = ax.get_ylim()
            ax.plot([med0, med0], yl, 'k')
            ax.plot([med1, med1], yl, 'r')
            
            ax.legend()
            ax.set_xticks(np.log(xs))
            ax.set_xticklabels([str(x) for x in xs])
            ax.set_yticks(np.arange(0, 0.25, 0.1))
            ax.set_ylabel('Probability')
            ax.set_xlabel('Time to next resp (s)')
            ax.set_title(f'Bin {i + 1}')
            cleanFigure(ax)
            
            # Calculate statistics
            d_tunp[i] = ((np.nanmedian(np.exp(tunp_all_log[ind0])) - 
                         np.nanmedian(np.exp(tunp_all_log[ind1]))) / 
                        np.nanmedian(np.exp(tunp_all_log[ind1])) * 100)
            
            tunp0.append(tunp_all_log[ind0])
            tunp1.append(tunp_all_log[ind1])
            
            valid0 = tunp_all_log[ind0][~np.isnan(tunp_all_log[ind0])]
            valid1 = tunp_all_log[ind1][~np.isnan(tunp_all_log[ind1])]
            if len(valid0) > 1 and len(valid1) > 1:
                _, p_tunp[i] = ttest_ind(valid0, valid1)
            
            choices0.append(choice_all_sel[ind0])
            choices1.append(choice_all_sel[ind1])
            p_switch0[i] = np.nanmean(choice_all_sel[ind0] == 1)
            p_switch1[i] = np.nanmean(choice_all_sel[ind1] == 1)
            
            ticks.append(f'{var_bins[i]:.1f} - {var_bins[i+1]:.1f}')
        
        # Choice bar plot
        ax = axes[n_bins]
        for i in range(n_bins):
            stay_frac = np.nanmean(choices0[i] == 0) * 100 if len(choices0[i]) > 0 else 0
            switch_frac = np.nanmean(choices0[i] == 1) * 100 if len(choices0[i]) > 0 else 0
            ax.bar(i + 1, stay_frac, label='Stay' if i == 0 else '')
            ax.bar(i + 1, switch_frac, bottom=stay_frac, label='Switch' if i == 0 else '')
        
        ax.set_xlabel('Bins')
        ax.set_ylabel('%')
        ax.legend()
        cleanFigure(ax)
        
        # Bin distribution
        ax = axes[n_bins + 1]
        ax.bar(range(len(var_bins) - 1), np.diff(var_bins))
        cleanFigure(ax)
        
        plt.tight_layout()
        plt.show()
        
        # Calculate AUC
        AUC = np.zeros(n_bins)
        AUC_ci = np.zeros((n_bins, 2))
        
        for i in range(n_bins):
            if len(tunp0[i]) > 0 and len(tunp1[i]) > 0:
                y_true, y_score = format_by_class(
                    tunp0[i][~np.isnan(tunp0[i])], 
                    tunp1[i][~np.isnan(tunp1[i])]
                )
                AUC[i], AUC_ci[i] = auc(y_true, y_score, method='boot', n_boot=2000)
        
        print(f"d_tunp: {d_tunp}")
        print(f"AUC: {AUC}")
        print(f"AUC CI: {np.nanmean(np.abs(AUC_ci - AUC.reshape(-1, 1)), axis=1)}")
        
        return d_tunp, AUC, p_tunp
    
    # =========================================================================
    # SECTION: Decorrelation results summary (Fig. S4c)
    # =========================================================================
    
    def plot_decorrelation_summary(self, decor_file):
        """
        Plot summary of decorrelation results.
        Corresponds to MATLAB section: summary of decorrelation results
        
        Parameters:
        -----------
        decor_file : str
            Path to decorrelation matrices file
        """
        data = load_mat_file(decor_file)
        
        irel_param_matrix = data.get('irelParamMatrix', {})
        neur_matrix = data.get('neurMatrix', {})
        o1_neur_matrix = data.get('o1NeurMatrix', {})
        
        irel_corr = []
        o_irel_corr = []
        o_irel_pval = []
        
        for s in self.valid_sessions:
            if s not in irel_param_matrix:
                continue
            
            X = irel_param_matrix[s]
            Y = neur_matrix[s]
            oY = o1_neur_matrix[s]
            
            # Find valid indices
            combined = np.column_stack([X, Y, oY])
            i_invalid = np.where(np.isnan(combined.sum(axis=1)) | 
                                np.isinf(combined.sum(axis=1)))[0]
            i_valid = np.setdiff1d(np.arange(X.shape[0]), i_invalid)
            
            if len(i_valid) < 3:
                continue
            
            # Correlation before decorrelation
            R = np.corrcoef(np.column_stack([Y[i_valid], X[i_valid]]).T)
            irel_corr.append(R[-3:, :-3])
            
            # Correlation after decorrelation
            R_o = np.corrcoef(np.column_stack([oY[i_valid], X[i_valid]]).T)
            o_irel_corr.append(R_o[-3:, :-3])
        
        if len(irel_corr) == 0:
            print("No valid sessions for decorrelation analysis")
            return
        
        irel_corr = np.hstack(irel_corr)
        o_irel_corr = np.hstack(o_irel_corr)
        
        # Plot
        fig, axes = plt.subplots(3, 1, figsize=(10, 12))
        
        for i in range(3):
            ax = axes[i]
            
            ax.hist(irel_corr[i, :], bins=np.arange(-0.5, 0.52, 0.02),
                   color='r', alpha=0.5, edgecolor='r', label='Before')
            ax.hist(o_irel_corr[i, :], bins=np.arange(-0.5, 0.52, 0.01),
                   color='k', alpha=0.5, edgecolor='k', label='After')
            
            cleanFigure(ax)
        
        axes[2].set_ylabel('Neurons')
        axes[2].set_xlabel('Correlation coefficient')
        axes[0].legend()
        
        plt.tight_layout()
        plt.show()
        
        return irel_corr, o_irel_corr
    
    # =========================================================================
    # SECTION: PSTH for tslp and lrr (Fig. 3a)
    # =========================================================================
    
    def plot_psth_tslp_lrr(self, session_idx=79, unit=6, variable='tslp'):
        """
        Plot PSTH for time since last press or local reward rate.
        Corresponds to MATLAB section: psth for tslp and lrr
        
        Parameters:
        -----------
        session_idx : int
            Session index
        unit : int
            Unit index to plot
        variable : str
            Variable to condition on ('tslp' or 'lrr')
        """
        s = session_idx
        
        # Load data
        data = self.load_session_data(s, 'binnedFr200_pre3post2')
        
        if variable == 'tslp':
            X = np.log(self.tslp[s][self.i_forag[s]])
            cl = tslpCl
        else:
            X = self.rew_ratio[s][self.i_forag[s]]
            cl = lrrCl
        
        # Get firing rate data
        t_fr = data.get('tFr', None)
        if t_fr is None:
            print("No tFr data found")
            return
        
        # Find low and high percentile trials
        ind_L = np.where(X <= np.percentile(X, 20))[0]
        ind_H = np.where(X >= np.percentile(X, 80))[0]
        
        u = unit
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # High resolution PSTH
        ax = axes[0, 0]
        
        i_forag = self.i_forag[s]
        
        m1, e1 = mWe(np.squeeze(t_fr[i_forag[ind_L], :]), 0)
        m2, e2 = mWe(np.squeeze(t_fr[i_forag[ind_H], :]), 0)
        
        x = np.arange(-3000, 2001)
        x_mark = np.arange(-1500, 1600, 500)
        i_show = np.arange(0, len(x), 10)
        
        jbfill(x[i_show], m1[i_show] + e1[i_show], m1[i_show] - e1[i_show], 
               [0.4, 0.4, 0.4], alpha=0.4, ax=ax)
        jbfill(x[i_show], m2[i_show] + e2[i_show], m2[i_show] - e2[i_show],
               cl, alpha=0.4, ax=ax)
        ax.plot(x[i_show], m1[i_show], color=[0.4, 0.4, 0.4], label='Low prob')
        ax.plot(x[i_show], m2[i_show], color=cl, label='High prob')
        
        ax.set_xlim([-1800, 1800])
        ax.set_title(f's={s}, u={u}')
        ax.set_xlabel('Time to response (ms)')
        ax.set_ylabel('Firing rate (sp/s)')
        ax.set_xticks(x_mark)
        ax.legend()
        cleanFigure(ax)
        
        plt.tight_layout()
        plt.show()
    
    # =========================================================================
    # SECTION: Population prediction (Fig. 3c)
    # =========================================================================
    
    def plot_population_prediction(self, pop_pred_file):
        """
        Plot population prediction results.
        Corresponds to MATLAB section: plotting the prediction results
        
        Parameters:
        -----------
        pop_pred_file : str
            Path to population prediction results file
        """
        data = load_mat_file(pop_pred_file)
        
        pop_pred_tslp = data.get('popPredTslp', {})
        pop_pred_lrr = data.get('popPredLrr', {})
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        m_gof_tslp_all = {}
        m_gof_lrr_all = {}
        
        for s in self.valid_sessions:
            if s not in pop_pred_tslp:
                continue
            
            # Get neuron counts
            ns = [1] + list(range(5, pop_pred_tslp[s].shape[0] + 1, 5))
            
            m1 = np.zeros(len(ns))
            e1 = np.zeros(len(ns))
            m2 = np.zeros(len(ns))
            e2 = np.zeros(len(ns))
            
            for i, n in enumerate(ns):
                if n <= pop_pred_tslp[s].shape[0]:
                    gof_tslp = [pop_pred_tslp[s][n-1, j].get('gofTest', np.nan) 
                               for j in range(pop_pred_tslp[s].shape[1])]
                    gof_lrr = [pop_pred_lrr[s][n-1, j].get('gofTest', np.nan)
                              for j in range(pop_pred_lrr[s].shape[1])]
                    
                    m1[i], e1[i] = mWe(gof_tslp, 0)
                    m2[i], e2[i] = mWe(gof_lrr, 0)
                    
                    m_gof_tslp_all[s, n] = np.nanmean(gof_tslp)
                    m_gof_lrr_all[s, n] = np.nanmean(gof_lrr)
            
            # Plot tslp
            ax = axes[0]
            color = tslpCl / 2 if s in self.gin_sessions else tslpCl
            ax.errorbar(ns, m1, yerr=e1, color=color, linewidth=2, capsize=0)
            
            # Plot lrr
            ax = axes[1]
            color = lrrCl / 2 if s in self.gin_sessions else lrrCl
            ax.errorbar(ns, m2, yerr=e2, color=color, linewidth=2, capsize=0)
        
        axes[0].set_xlabel('No. of Neurons')
        axes[0].set_ylabel('Decoder perf (r)')
        axes[0].set_title('Time since last press')
        cleanFigure(axes[0])
        
        axes[1].set_title('Local reward rate')
        cleanFigure(axes[1])
        
        plt.tight_layout()
        plt.show()
        
        return m_gof_tslp_all, m_gof_lrr_all
    
    # =========================================================================
    # SECTION: GLM contribution analysis (Fig. S6b)
    # =========================================================================
    
    def plot_glm_contribution(self, glm_files):
        """
        Plot contribution of reward dynamics in GLM performance.
        Corresponds to MATLAB section: contribution of rew dyn in GLM's performance
        
        Parameters:
        -----------
        glm_files : dict
            Dictionary mapping extension names to file paths
        """
        ext = ['All', 'tslp', 'lrr', 'base']
        glm_corr_coef = {}
        
        for l, ext_name in enumerate(ext):
            if ext_name not in glm_files:
                continue
            
            data = load_mat_file(glm_files[ext_name])
            GLM_cont = data.get('GLMcont', {})
            
            m = 0
            glm_corr_coef[l] = []
            
            for s in self.valid_sessions:
                if s not in GLM_cont:
                    continue
                
                for u in range(len(GLM_cont[s])):
                    r_test = GLM_cont[s][u].get('rTest', np.nan)
                    glm_corr_coef[l].append(np.nanmedian(r_test))
                    m += 1
            
            glm_corr_coef[l] = np.array(glm_corr_coef[l])
        
        if len(ext) - 1 not in glm_corr_coef or len(glm_corr_coef[len(ext) - 1]) == 0:
            print("Base GLM not found")
            return
        
        base = glm_corr_coef[len(ext) - 1]
        
        dr_tslp = (glm_corr_coef.get(1, base) - base) / base * 100
        dr_lrr = (glm_corr_coef.get(2, base) - base) / base * 100
        dr_all = (glm_corr_coef.get(0, base) - base) / base * 100
        
        bins = np.arange(-50, 110, 10)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        N1, _ = np.histogram(dr_tslp, bins=bins)
        N1 = N1 / np.sum(N1)
        ax.plot(bins[:-1], N1, color=[0, 0.6, 0], linewidth=2, label='tslp')
        
        N2, _ = np.histogram(dr_lrr, bins=bins)
        N2 = N2 / np.sum(N2)
        ax.plot(bins[:-1], N2, color=[0, 0, 1], linewidth=2, label='lrr')
        
        N3, _ = np.histogram(dr_all, bins=bins)
        N3 = N3 / np.sum(N3)
        ax.plot(bins[:-1], N3, color=[0, 0, 0], linewidth=2, label='all')
        
        ax.set_xlabel('Inc. in correlation (%)')
        ax.set_ylabel('Frac. of neurons')
        ax.set_ylim([0, 1])
        ax.legend()
        cleanFigure(ax)
        
        plt.tight_layout()
        plt.show()
        
        print(f"Median improvement - tslp: {np.nanmedian(dr_tslp):.2f}%, lrr: {np.nanmedian(dr_lrr):.2f}%")
        
        # Statistical tests
        p = np.zeros(2)
        p[0] = wilcoxon(dr_tslp[~np.isnan(dr_tslp)] - 0.01, alternative='greater').pvalue
        p[1] = wilcoxon(dr_lrr[~np.isnan(dr_lrr)] - 0.01, alternative='greater').pvalue
        
        h, th, q = fdr_bh(p)
        print(f"FDR-corrected q-values: {q}")
        
        return dr_tslp, dr_lrr, dr_all
    
    # =========================================================================
    # SECTION: Decoding performance comparison (Fig. 5b-d)
    # =========================================================================
    
    def plot_decoding_comparison(self, pred_files, bin_size=200):
        """
        Plot decoding performance comparison across methods.
        Corresponds to MATLAB section: decoding rew/choice/tunp from different things
        
        Parameters:
        -----------
        pred_files : dict
            Dictionary with paths to prediction result files
        bin_size : int
            Bin size in ms
        """
        # Load data
        data_prew = load_mat_file(pred_files.get('pRew', ''))
        data_pred = load_mat_file(pred_files.get('pred', ''))
        data_pop = load_mat_file(pred_files.get('pop', ''))
        
        n_points = 3
        range_idx = np.arange(11, 22)
        
        sel_sessions = self.valid_sessions
        
        # Initialize storage
        m_behav_pred_rew = {}
        m_neur_pred_rew = {}
        m_pop_pred_rew = {}
        m_behav_pred_choice = {}
        m_neur_pred_choice = {}
        m_pop_pred_choice = {}
        m_behav_pred_tunp = {}
        m_neur_pred_tunp = {}
        m_pop_pred_tunp = {}
        m_prew_pred_rew = {}
        
        ts = data_pred.get('ts', np.arange(-20, 21))
        
        for s in sel_sessions:
            # Extract prediction results for this session
            behav_pred_rew = data_pred.get(f'behavCompPredRew_{s}', None)
            neur_pred_rew = data_pred.get(f'neurCompPredRew_{s}', None)
            pop_pred_rew = data_pop.get(f'popPredRew_{s}', None)
            
            if behav_pred_rew is None:
                continue
            
            n_cuts = data_pred.get(f'nCuts_{s}', [0])
            
            for ch in range(len(n_cuts) - 1):
                br = np.zeros(len(ts))
                nr = np.zeros(len(ts))
                pr = np.zeros(len(ts))
                
                for t in range(len(ts)):
                    br[t] = behav_pred_rew[t][ch].get('aucTest', [0.5])[0] - 0.5 + 0.5
                    nr[t] = neur_pred_rew[t][ch].get('aucTest', [0.5])[0] - 0.5 + 0.5
                    pr[t] = pop_pred_rew[t].get('aucTest', [0.5])[ch] - 0.5 + 0.5
                
                # Get top n_points
                tmp = np.sort(br[range_idx])
                m_behav_pred_rew[s, ch] = np.nanmean(tmp[-n_points:])
                
                tmp = np.sort(nr[range_idx])
                m_neur_pred_rew[s, ch] = np.nanmean(tmp[-n_points:])
                
                tmp = np.sort(pr[range_idx])
                m_pop_pred_rew[s, ch] = np.nanmean(tmp[-n_points:])
        
        # Aggregate across chunks
        mbr = np.array([np.nanmean([m_behav_pred_rew.get((s, ch), np.nan) 
                                    for ch in range(10)]) for s in sel_sessions])
        mnr = np.array([np.nanmean([m_neur_pred_rew.get((s, ch), np.nan)
                                    for ch in range(10)]) for s in sel_sessions])
        mpr = np.array([np.nanmean([m_pop_pred_rew.get((s, ch), np.nan)
                                    for ch in range(10)]) for s in sel_sessions])
        
        scl = 0.4
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Reward decoding
        ax = axes[0]
        for i, s in enumerate(sel_sessions):
            ax.plot(1 + jit(1), mbr[i], '.', markersize=10, color=[0.5, 0.5, 0.5])
            ax.plot(2 + jit(1), mnr[i], '.', markersize=10, color=[0, 0.5, 0.5])
            ax.plot(3 + jit(1), mpr[i], '.', markersize=10, color=[0.5, 0, 0.5])
        
        ax.plot([0.5, 1.5], [np.nanmean(mbr)] * 2, color=[0.5, 0.5, 0.5], linewidth=2)
        ax.plot([1.5, 2.5], [np.nanmean(mnr)] * 2, color=[0, 0.5, 0.5], linewidth=2)
        ax.plot([2.5, 3.5], [np.nanmean(mpr)] * 2, color=[0.5, 0, 0.5], linewidth=2)
        
        ax.set_xticks([1, 2, 3])
        ax.set_xticklabels(['Task vars', 'Neur repres', 'Population'])
        ax.set_ylabel('Perf. (AUC)')
        ax.set_title('Reward')
        ax.set_ylim([0.5, 0.95])
        cleanFigure(ax)
        
        # Print statistics
        print(f"Mean reward decoding - Task: {np.nanmean(mbr):.3f}, "
              f"Neural: {np.nanmean(mnr):.3f}, Pop: {np.nanmean(mpr):.3f}")
        
        # Statistical tests
        p = np.zeros(3)
        valid_mbr = mbr[~np.isnan(mbr)]
        valid_mnr = mnr[~np.isnan(mnr)]
        valid_mpr = mpr[~np.isnan(mpr)]
        
        if len(valid_mbr) > 0 and len(valid_mnr) > 0:
            p[0] = wilcoxon(valid_mbr, valid_mnr).pvalue
        if len(valid_mbr) > 0 and len(valid_mpr) > 0:
            p[1] = wilcoxon(valid_mbr, valid_mpr).pvalue
        if len(valid_mnr) > 0 and len(valid_mpr) > 0:
            p[2] = wilcoxon(valid_mnr, valid_mpr).pvalue
        
        h, th, q = fdr_bh(p)
        print(f"FDR-corrected q-values: {q}")
        
        plt.tight_layout()
        plt.show()
        
        return mbr, mnr, mpr
    
    # =========================================================================
    # SECTION: Time-resolved decoding (Fig. 5b-d left)
    # =========================================================================
    
    def plot_time_resolved_decoding(self, pred_files, session_idx, bin_size=200):
        """
        Plot time-resolved decoding performance.
        Corresponds to MATLAB section: example sessions of prediction results
        
        Parameters:
        -----------
        pred_files : dict
            Dictionary with paths to prediction result files
        session_idx : int
            Session index to plot
        bin_size : int
            Bin size in ms
        """
        s = session_idx
        
        # Load data
        data_prew = load_mat_file(pred_files.get('pRew', ''))
        data_pred = load_mat_file(pred_files.get('pred', ''))
        data_pop = load_mat_file(pred_files.get('pop', ''))
        
        ts = data_pred.get('ts', np.arange(-20, 21))
        n_cuts = data_pred.get(f'nCuts_{s}', [0, 100])
        
        sm_win = 2
        
        # Extract prediction results
        behav_pred_rew = data_pred.get(f'behavCompPredRew_{s}', None)
        neur_pred_rew = data_pred.get(f'neurCompPredRew_{s}', None)
        pop_pred_rew = data_pop.get(f'popPredRew_{s}', None)
        
        if behav_pred_rew is None:
            print(f"No prediction data for session {s}")
            return
        
        n_chunks = len(n_cuts) - 1
        
        br = np.zeros((n_chunks, len(ts)))
        nr = np.zeros((n_chunks, len(ts)))
        pr = np.zeros((n_chunks, len(ts)))
        
        for ch in range(n_chunks):
            for t in range(len(ts)):
                br[ch, t] = behav_pred_rew[t][ch].get('aucTest', [0.5])[0] - 0.5 + 0.5
                nr[ch, t] = neur_pred_rew[t][ch].get('aucTest', [0.5])[0] - 0.5 + 0.5
                pr[ch, t] = pop_pred_rew[t].get('aucTest', [0.5])[ch] - 0.5 + 0.5
            
            br[ch, :] = nanfastsmooth(br[ch, :], sm_win)
            nr[ch, :] = nanfastsmooth(nr[ch, :], sm_win)
            pr[ch, :] = nanfastsmooth(pr[ch, :], sm_win)
        
        # Find valid chunks
        ind = np.where(np.diff(n_cuts) > 0)[0]
        
        e_opac = 0.05
        time_interval = ts * bin_size / 1000
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        m1, e1 = mWe(br[ind, :], 0)
        m2, e2 = mWe(nr[ind, :], 0)
        m3, e3 = mWe(pr[ind, :], 0)
        
        jbfill(time_interval, m1 + e1, m1 - e1, [0, 0, 0], alpha=e_opac, ax=ax)
        jbfill(time_interval, m2 + e2, m2 - e2, [0, 0.5, 0.5], alpha=e_opac, ax=ax)
        jbfill(time_interval, m3 + e3, m3 - e3, [0.5, 0, 0.5], alpha=e_opac, ax=ax)
        
        ax.plot(time_interval, m1, color=[0, 0, 0], linewidth=2, label='Task variables')
        ax.plot(time_interval, m2, color=[0, 0.5, 0.5], linewidth=2, label='Neural repres')
        ax.plot(time_interval, m3, color=[0.5, 0, 0.5], linewidth=2, label='Population')
        
        ax.set_xlim([-3, 1])
        ax.set_ylabel('Performance (AUC)')
        ax.set_xlabel('Time to press (s)')
        ax.set_title(f's = {s}, Reward')
        ax.legend()
        cleanFigure(ax)
        
        plt.tight_layout()
        plt.show()


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """
    Main function to run the analysis.
    Modify paths and parameters as needed.
    """
    # Initialize analysis
    proj_path = ''  # Set your project path here
    analysis = NeuralAnalysis(proj_path)
    
    print("Neural Analysis Script")
    print("=" * 50)
    print("\nThis script contains the following analysis sections:")
    print("1. Sample units encoding (plot_sample_units)")
    print("2. Reward predictor correlation (plot_reward_predictor_correlation)")
    print("3. Reward prediction (plot_reward_prediction)")
    print("4. Predictor effects on choices (plot_predictor_effects)")
    print("5. Reward/wait effects (plot_reward_wait_effect)")
    print("6. Decorrelation summary (plot_decorrelation_summary)")
    print("7. PSTH for tslp/lrr (plot_psth_tslp_lrr)")
    print("8. Population prediction (plot_population_prediction)")
    print("9. GLM contribution (plot_glm_contribution)")
    print("10. Decoding comparison (plot_decoding_comparison)")
    print("11. Time-resolved decoding (plot_time_resolved_decoding)")
    print("\nTo run analyses, load your data and call the appropriate methods.")
    print("\nExample usage:")
    print("  analysis = NeuralAnalysis('/path/to/project/')")
    print("  data = load_mat_file('/path/to/data.mat')")
    print("  analysis.plot_sample_units(session_idx=74, units=[15, 17, 19])")
    

if __name__ == "__main__":
    main()