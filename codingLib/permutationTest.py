import numpy as np
import itertools
import math
import matplotlib.pyplot as plt
from scipy.stats import t
import warnings

def permutationTest(sample1, sample2, permutations, sidedness='both', exact=0, plotresult=0, showprogress=0):
    """
    MATLAB: function [p, observeddifference, effectsize] = permutationTest(sample1, sample2, permutations, varargin)
    
    Permutation test (aka randomisation test), testing for a difference 
    in means between two samples.
    
    Args:
        sample1 (np.ndarray): Vector of measurements from one (experimental) sample.
        sample2 (np.ndarray): Vector of measurements from a second (control) sample.
        permutations (int): The number of permutations (ignored if exact=1).
        sidedness (str): 'both' (two-sided, default), 'smaller', or 'larger' (one-sided).
        exact (int): Whether to run an exact test (1|0, default 0).
        plotresult (int): Whether or not to plot the distribution (1|0, default 0).
        showprogress (int): Not implemented/ignored in this Python script.
        
    Returns:
        tuple: (p, observeddifference, effectsize) - p-value, observed mean difference, Hedges' g.
    """

    # --- 1. Input Parsing and Setup ---
    sample1 = np.array(sample1).flatten()
    sample2 = np.array(sample2).flatten()
    
    # Remove NaNs from samples before combining, as per nanmean usage
    sample1 = sample1[~np.isnan(sample1)]
    sample2 = sample2[~np.isnan(sample2)]
    
    n1 = len(sample1)
    n2 = len(sample2)
    
    if n1 == 0 or n2 == 0:
        warnings.warn("One or both samples are empty after removing NaNs.")
        return np.nan, np.nan, np.nan

    allobservations = np.concatenate([sample1, sample2])
    N_total = len(allobservations)
    
    # Observed difference: mean(sample1) - mean(sample2)
    # MATLAB uses nanmean, which is necessary if NaNs were not removed beforehand.
    observeddifference = np.mean(sample1) - np.mean(sample2)

    # --- Hedges' g Effect Size ---
    # Pooled standard deviation (unbiased estimator)
    # std(sample)^2 uses N-1 in the denominator by default in MATLAB/NumPy (ddof=1)
    pooledstd = np.sqrt(
        ( (n1 - 1) * np.std(sample1, ddof=1)**2 + (n2 - 1) * np.std(sample2, ddof=1)**2 ) / (N_total - 2)
    )
    
    if pooledstd == 0:
        effectsize = np.nan # Undefined effect size if variance is zero
    else:
        # Hedges' g = observeddifference / pooledstd
        effectsize = observeddifference / pooledstd
        
    # --- Exact Test Pre-checks ---
    
    # nchoosek(N_total, n1)
    try:
        max_combinations = math.comb(N_total, n1)
    except AttributeError:
        # Fallback for older Python/math versions
        max_combinations = 1
        
    if not exact and permutations > max_combinations:
        warnings.warn(
            f"The number of permutations ({permutations}) is higher than the number of possible combinations ({max_combinations})."
            "Consider running an exact test using the 'exact' argument."
        )

    # --- 2. Running Test ---
    
    randomdifferences = []
    
    if exact:
        # Get all possible combinations (indices for the first sample)
        # MATLAB: allcombinations = nchoosek(1:numel(allobservations), numel(sample1));
        all_indices = range(N_total)
        allcombinations_indices = list(itertools.combinations(all_indices, n1))
        
        permutations = len(allcombinations_indices)
        
        for combo_indices in allcombinations_indices:
            # combo_indices is the index set for the first sample
            combo_mask = np.zeros(N_total, dtype=bool)
            combo_mask[list(combo_indices)] = True
            
            randomSample1 = allobservations[combo_mask]
            randomSample2 = allobservations[~combo_mask]
            
            randomdifferences.append(np.mean(randomSample1) - np.mean(randomSample2))
            
    else:
        # Random Permutation
        for n in range(permutations):
            # MATLAB: permutation = randperm(length(allobservations));
            permutation = np.random.permutation(N_total)
            
            # Dividing into two samples
            randomSample1 = allobservations[permutation[:n1]]
            randomSample2 = allobservations[permutation[n1:]]
            
            randomdifferences.append(np.mean(randomSample1) - np.mean(randomSample2))

    randomdifferences = np.array(randomdifferences)

    # --- 3. Calculating P-value ---
    
    # MATLAB uses: (length(find(...) ) + 1) / (permutations + 1) 
    # to avoid zero p-values (recommended practice).
    
    if sidedness == 'both':
        # p = (length(find(abs(randomdifferences) > abs(observeddifference)))+1) / (permutations+1);
        count = np.sum(np.abs(randomdifferences) >= np.abs(observeddifference))
        p = (count + 1) / (permutations + 1)
    elif sidedness == 'smaller':
        # p = (length(find(randomdifferences < observeddifference))+1) / (permutations+1);
        count = np.sum(randomdifferences <= observeddifference)
        p = (count + 1) / (permutations + 1)
    elif sidedness == 'larger':
        # p = (length(find(randomdifferences > observeddifference))+1) / (permutations+1);
        count = np.sum(randomdifferences >= observeddifference)
        p = (count + 1) / (permutations + 1)
    else:
        raise ValueError("Invalid 'sidedness' argument. Must be 'both', 'smaller', or 'larger'.")

    # --- 4. Plotting Result ---
    if plotresult:
        plt.figure()
        
        # MATLAB: hist/histogram(randomdifferences, 20);
        # Use Matplotlib's hist/histrogram which works well across versions
        plt.hist(randomdifferences, bins=20, color='skyblue', edgecolor='black')
        plt.xlabel('Random Differences')
        plt.ylabel('Count')
        plt.title('Permutation Test Distribution')
        
        # Plot observed difference
        # Need to estimate the height for the observed difference marker
        # We can use a small vertical line or estimate max height from the hist output
        
        # Re-run hist to get bar heights
        counts, bins, _ = plt.hist(randomdifferences, bins=20)
        max_count = np.max(counts) if len(counts) > 0 else 0
        
        # Find the bin corresponding to the observed difference
        bin_index = np.digitize(observeddifference, bins) - 1
        marker_height = counts[bin_index] * 1.05 if bin_index < len(counts) else max_count * 1.05
        
        # plot(observeddifference, 0, '*r', ...)
        # Plotting the star at y=0 is not very visual. We use a vertical line.
        od = plt.plot([observeddifference, observeddifference], [0, marker_height], 'r--', 
                      linewidth=2, label='Observed Difference')
        
        plt.plot(observeddifference, marker_height, 'r*', markersize=10) # Star marker
        
        plt.legend(handles=od, title=f"Effect size: {effectsize:.2f}\np = {p:.6f}")
        plt.show()

    return p, observeddifference, effectsize