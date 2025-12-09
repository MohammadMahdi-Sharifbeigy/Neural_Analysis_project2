import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon, mannwhitneyu
import pandas as pd

# --- Helper function: fdr_bh (Benjamini-Hochberg FDR correction) ---
def fdr_bh(p_values):
    """
    MATLAB: [h,th,q] = fdr_bh(p_values)
    
    Benjamini-Hochberg procedure for controlling the False Discovery Rate (FDR).
    
    Args:
        p_values (list or np.ndarray): List of p-values.
        
    Returns:
        tuple: (h, th, q) - 
               h (bool array): A boolean array indicating which p-values are significant 
                               after correction (threshold at 0.05).
               th (float): The FDR-corrected significance threshold.
               q (np.ndarray): The corrected p-values (q-values).
    """
    p_values = np.array(p_values)
    n = len(p_values)
    
    # Remove NaN values before sorting and correction
    valid_indices = ~np.isnan(p_values)
    p_values_valid = p_values[valid_indices]
    n_valid = len(p_values_valid)
    
    if n_valid == 0:
        return np.zeros(n, dtype=bool), 0.05, np.full(n, np.nan)

    # 1. Sort the p-values in ascending order
    sorted_p_values = np.sort(p_values_valid)
    
    # 2. Calculate the BH critical value
    # Formula: (i / m) * alpha, where m is the number of tests (n_valid)
    i = np.arange(1, n_valid + 1)
    alpha = 0.05
    critical_values = (i / n_valid) * alpha
    
    # 3. Find the largest k such that p_k <= (k / m) * alpha
    # Use flip and cummax to find the largest p_i that satisfies the condition,
    # then set all smaller p_j (j < i) to the same value.
    q_values_valid = np.zeros(n_valid)
    
    # The q-value is min(p_i * m / i, min(q_{i+1}, ..., q_m))
    # This is equivalent to: (p_i * m / i) in ascending order, 
    # then taking the cumulative minimum from the end.
    
    # Calculate initial q-values: (p_i * m / i)
    q_values_valid = sorted_p_values * n_valid / i
    
    # Enforce monotonicity: q_i must be <= q_{i+1}
    q_values_valid = np.minimum.accumulate(q_values_valid[::-1])[::-1]
    
    # Cap q-values at 1
    q_values_valid[q_values_valid > 1.0] = 1.0
    
    # Map back to original order, including NaNs
    q = np.full(n, np.nan)
    # The sort order indices are not needed for this q-value implementation, 
    # but we need to match the valid p-values to their original positions.
    # The original MATLAB code only returns the array `q` which contains 
    # all p1 and p2 values corrected. We must put them back in the order [p1_corrected, p2_corrected].
    
    # We will simply return the corrected values for the valid ones in the sorted order
    # and expect the main function to handle the original order.
    
    # Find the largest p-value that is less than or equal to its critical value
    # This determines the threshold 'th' and the significance 'h'
    is_significant = sorted_p_values <= critical_values
    
    # Find the largest index k where the condition is met
    if np.any(is_significant):
        k = np.max(np.where(is_significant)[0])
        th = sorted_p_values[k] # This is a common definition for the threshold
        
        # The significance array 'h'
        # All p-values less than or equal to the threshold 'th' are significant.
        # Another common approach for q-values is: p_i * m / i <= alpha
        
        # We will use the explicit q-value definition for significance h
        h_valid = q_values_valid <= alpha
        th = np.max(sorted_p_values[h_valid]) if np.any(h_valid) else 0.0
    else:
        h_valid = np.zeros(n_valid, dtype=bool)
        th = 0.0
    
    # Reconstruct the output 'h' and 'q' in the original unsorted order
    # (Since we are returning [p1 p2] order, we just need to ensure the sorting 
    # is consistent for the whole set [p1 p2]).
    
    # To return q in the order [p1, p2], we need to know the original indices.
    # A simplified approach (matching the MATLAB output format for q):
    # This implementation is a direct BH correction on the input p-values array.
    
    # Create an array of sorted indices
    sorted_indices = np.argsort(p_values_valid)
    # Inverse sorted indices (where the sorted element came from)
    original_indices_valid = np.argsort(sorted_indices)

    # Re-order h_valid and q_values_valid back to the input order
    h_out = np.zeros(n, dtype=bool)
    q_out = np.full(n, np.nan)
    
    # The following mapping logic is simplified for the specific MATLAB 
    # output structure [p1 p2].
    # In the MATLAB script, it calls fdr_bh([p1 p2]), so p_values is already 
    # in the correct concatenated order.
    
    # A cleaner approach for the combined array [p1 p2]:
    
    # Sort the combined array, keeping track of indices
    p_combined = np.array(p_values)
    sorted_indices = np.argsort(p_combined)
    sorted_p_values = p_combined[sorted_indices]
    
    # Filter out NaNs
    valid_mask = ~np.isnan(sorted_p_values)
    p_values_valid = sorted_p_values[valid_mask]
    n_valid = len(p_values_valid)
    
    if n_valid == 0:
         return np.zeros(n, dtype=bool), 0.05, np.full(n, np.nan)

    # Re-calculate q-values
    i = np.arange(1, n_valid + 1)
    q_values_valid = p_values_valid * n_valid / i
    q_values_valid = np.minimum.accumulate(q_values_valid[::-1])[::-1]
    q_values_valid[q_values_valid > 1.0] = 1.0
    
    # Map q-values back to the original p_combined array order
    q_out = np.full(n, np.nan)
    q_out[sorted_indices[valid_mask]] = q_values_valid

    # Calculate h and th
    h_out = q_out <= alpha
    th = np.max(p_combined[h_out]) if np.any(h_out) else 0.0

    return h_out, th, q_out

def betterThanBarPlot(x, y, jitRange, cl, tl, yl, xLabel, yLabel):
    """
    MATLAB: function q = betterThanBarPlot(x, y, jitRange, cl, tl, yl, xLabel, yLabel)
    
    Plots scattered data with jitter and mean line, performs statistical tests,
    and returns FDR-corrected p-values (q-values).
    
    Args:
        x (list or np.ndarray): Group index/identifier.
        y (list or np.ndarray): Data values.
        jitRange (float): Range for the random jitter.
        cl (np.ndarray): Color array for groups (Nx3 matrix).
        tl (str): Title of the plot.
        yl (list or tuple): Y-axis limits [ymin, ymax].
        xLabel (list or tuple): Labels for the x-ticks.
        yLabel (str): Label for the y-axis.
        
    Returns:
        np.ndarray: The FDR-corrected p-values (q-values).
    """
    
    # Convert inputs to NumPy arrays
    x = np.array(x)
    y = np.array(y)
    
    # 1. Jitter setup
    # MATLAB: jit = jitRange*randn(1,100000);
    jit = jitRange * np.random.randn(100000)
    
    plt.figure()
    plt.hold = True # Mimic hold on
    
    # Get unique group identifiers
    # MATLAB: ns = unique(x);
    ns = np.unique(x)
    
    p1 = [] # p-values for signrank (against 0 or the median, MATLAB default is median=0)
    p2 = [] # p-values for ranksum (pairwise comparison)
    
    # 2. Plotting and Statistical Tests
    for n_idx, n_val in enumerate(ns):
        # MATLAB: ind = find(x == ns(n));
        ind = np.where(x == n_val)[0]
        y_group = y[ind]
        
        # Jitter: ns(n) + jit(1:length(ind))
        x_jittered = n_val + jit[:len(ind)]
        
        # Scatter plot (MATLAB: scatter(..., 'filled', 'MarkerFaceAlpha', 0.5))
        # Use cl[n_idx, :] for the color (must be an array of colors, like Nx3)
        plt.scatter(
            x_jittered, y_group, 
            s=30, 
            color=cl[n_idx, :], 
            alpha=0.5, 
            label=xLabel[n_idx] if n_idx < len(xLabel) else str(n_val)
        )
        
        # Plot mean line (MATLAB: nanmean(y(ind))*[1 1])
        mean_y = np.nanmean(y_group)
        # MATLAB: plot(ns(n) + 10*[-jitRange jitRange], ...)
        plt.plot(
            n_val + 10 * np.array([-jitRange, jitRange]), 
            [mean_y, mean_y], 
            linewidth=3, 
            color=cl[n_idx, 0] # Using only the first channel for color intensity
        )
        
        # MATLAB: cleanFigure (a custom function, skipping direct conversion)
        
        # MATLAB: title(tl)
        plt.title(tl)
        
        # Statistical Tests
        # MATLAB: p1(n) = signrank(y(ind)); -> Wilcoxon Signed-Rank test (often against a median of 0)
        # For a single sample, wilcoxon compares against 0.
        try:
            # We use 'exact' method, which is the default for wilcoxon in older scipy versions
            # and is generally robust.
            p1_val = wilcoxon(y_group, alternative='two-sided').pvalue 
            p1.append(p1_val)
        except ValueError:
            # wilcoxon fails if all values are zero or not enough data points
            p1.append(np.nan)

        if n_idx > 0:
            # MATLAB: p2(n-1) = ranksum(y(x== ns(n-1)), y(ind)); -> Mann-Whitney U test (non-parametric two-sample test)
            y_prev_group = y[np.where(x == ns[n_idx - 1])[0]]
            
            try:
                # mannwhitneyu is two-sided by default. 'ranksum' is often used for it.
                p2_val = mannwhitneyu(y_prev_group, y_group, alternative='two-sided').pvalue
                p2.append(p2_val)
            except ValueError:
                p2.append(np.nan)
    
    # 3. Figure properties
    # MATLAB: xlim([ns(1)-1 ns(end)+1])
    plt.xlim(ns[0] - 1, ns[-1] + 1)
    
    # MATLAB: ylim(yl)
    plt.ylim(yl)
    
    # MATLAB: set(gca, 'XTick', ns), set(gca, 'XTickLabels', xLabel)
    plt.xticks(ns, xLabel)
    
    # MATLAB: ylabel(yLabel)
    plt.ylabel(yLabel)
    
    # 4. FDR Correction
    # MATLAB: [h,th,q] = fdr_bh([p1 p2]);
    p_combined = p1 + p2
    
    # NOTE: The fdr_bh implementation above is a custom function.
    # In a real environment, you'd ensure it's correct or use a library 
    # like statsmodels. Here, we use the custom implementation:
    h, th, q = fdr_bh(p_combined) 
    
    plt.show() # Display the plot
    
    # MATLAB returns 'q' as the only output: return q
    return q

# Example usage (for testing):
# np.random.seed(42)
# x_test = np.array([1, 1, 1, 2, 2, 2, 3, 3, 3])
# y_test = np.array([1.2, 1.5, 0.8, 2.5, 2.0, 3.1, 4.0, 3.8, 4.5])
# jitRange_test = 0.1
# cl_test = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]) # Red, Green, Blue colors
# tl_test = 'Sample Jitter Plot'
# yl_test = [0, 5]
# xLabel_test = ['Group A', 'Group B', 'Group C']
# yLabel_test = 'Data Value'
# 
# q_values = betterThanBarPlot(x_test, y_test, jitRange_test, cl_test, tl_test, yl_test, xLabel_test, yLabel_test)
# print(f"FDR-corrected p-values (q-values): {q_values}")