import numpy as np
from sklearn.model_selection import KFold
from sklearn.svm import SVC, SVR
from sklearn.metrics import accuracy_score, roc_auc_score, r2_score
from scipy.stats import pearsonr
from time import time as time_now
import warnings

# --- Utility Placeholder Functions (Must be defined elsewhere for full fidelity) ---

def corrcoeff(x, y):
    """Placeholder for the custom corrcoeff which handles NaNs and Inf."""
    x = np.array(x).flatten()
    y = np.array(y).flatten()
    
    valid_mask = np.isfinite(x) & np.isfinite(y)
    
    if np.sum(valid_mask) < 2:
        return np.nan
    
    r, _ = pearsonr(x[valid_mask], y[valid_mask])
    return r

def nanmean(data, axis=None):
    """Wrapper for nanmean."""
    return np.nanmean(data, axis=axis)

def nansum(data):
    """Wrapper for nansum (used for counting correct predictions)."""
    return np.nansum(data)

def bootSVM(X_train, Y_train, bootN, kernel):
    """
    Placeholder for bootSVM (Bootstrapped SVM classifier).
    Returning a standard SVC model as a proxy.
    """
    warnings.warn("Using SVC as a proxy for bootSVM. Actual implementation required.")
    
    # SVC requires binary classification labels to be 0 and 1, or -1 and 1
    # Assuming Y_train contains labels 0 and 1.
    if np.all(np.unique(Y_train) == np.array([1, 2])): # Handle common MATLAB/R 1 and 2
        Y_train = Y_train - 1 # Convert to 0 and 1
        
    # We ignore bootN for this placeholder, just train one RBF SVM
    model = SVC(kernel=kernel, probability=False, gamma='scale')
    model.fit(X_train, Y_train)
    return model

# GLM and BLR placeholders will use scipy/statsmodels equivalents below.
# CCA and PMD placeholders are complex and only approximated.

# --- Main Function ---

def chunkwiseCV(X, Y, iChunk, method, extras=None):
    """
    MATLAB: function results = chunkwiseCV(X, Y, iChunk, method, extras)
    
    Performs chunkwise cross-validation (leave-one-chunk-out CV) 
    using various statistical and machine learning methods.
    
    Args:
        X (np.ndarray): Predictor data (Features).
        Y (np.ndarray): Response data (Target).
        iChunk (np.ndarray): 1D array containing the starting indices of each chunk 
                             (1-based, mimicking MATLAB).
        method (str): The method to use ('SVM', 'SVMR', 'GLM', 'BLR', 'CCA', 'pCorr').
        extras (dict): Dictionary containing extra parameters like 'bootN', 'kernelScale', etc.
        
    Returns:
        dict: Dictionary containing results specific to the chosen method.
    """
    
    if extras is None:
        extras = {}
        
    # Ensure iChunk is 0-based for internal Python use
    # iChunk is typically 1-based indices in MATLAB: [1, N_chunk1+1, N_chunk1+N_chunk2+1, ...]
    # We use these indices to define the start and end of test/train indices below.
    iChunk = np.array(iChunk).flatten()
    
    results = {}
    N = len(Y)
    
    # --- Helper to define Test and Train Indices for the current chunk ---
    def get_indices(ch_idx):
        # MATLAB: testInd = iChunk(ch)+1:min(iChunk(ch+1), length(Y));
        # Test start is iChunk[ch] (1-based index) -> iChunk[ch] (0-based index)
        # Test end is min(iChunk[ch+1], N) (1-based index) -> min(iChunk[ch+1], N) (0-based exclusive index)
        
        # MATLAB indices are 1-based and inclusive.
        test_start_1based = iChunk[ch_idx] + 1
        test_end_1based = min(iChunk[ch_idx+1], N)
        
        # Python indices are 0-based and end-exclusive.
        test_start_0based = int(test_start_1based - 1)
        test_end_0based = int(test_end_1based)
        
        # Create test index array
        test_ind = np.arange(test_start_0based, test_end_0based)
        
        # Create training index array (everything not in test_ind)
        full_ind = np.arange(N)
        tr_ind = np.setdiff1d(full_ind, test_ind)
        
        return test_ind, tr_ind

    # --- Main Switch (Case) Statement ---
    
    if method == 'SVM':
        results['perCorr'] = np.zeros((len(iChunk) - 1, 3))
        results['perCorrTr'] = np.zeros((len(iChunk) - 1, 3))
        results['et'] = np.zeros(len(iChunk) - 1)
        results['predChoice'] = {}
        results['predChoiceTr'] = {}
        
        for ch in range(len(iChunk) - 1):
            t_start = time_now()
            test_ind, tr_ind = get_indices(ch)
            
            # Check for binary classification requirement (at least two unique classes)
            if len(np.unique(Y[tr_ind])) > 1:
                # MATLAB: svmModel = bootSVM(X(trInd,:),Y(trInd),extras.bootN, 'rbf');
                # Placeholder bootSVM returns a standard SVC model (labels assumed 0/1)
                svmModel = bootSVM(X[tr_ind, :], Y[tr_ind], extras.get('bootN', 10), extras.get('kernel', 'rbf'))
                
                # Predict (predict output labels 0 or 1)
                tmp_test = svmModel.predict(X[test_ind, :])
                tmp_tr = svmModel.predict(X[tr_ind, :])
                
                # Assuming MATLAB labels are 0 and 1 here (or converted to 0 and 1 in bootSVM)
                results['predChoice'][ch] = tmp_test
                results['predChoiceTr'][ch] = tmp_tr
                
                # Convert Y labels to 0/1 for comparison if they were originally 1/2
                Y_test_01 = Y[test_ind] if np.max(Y) <= 1 else Y[test_ind] - 1
                Y_tr_01 = Y[tr_ind] if np.max(Y) <= 1 else Y[tr_ind] - 1
                
                # Calculate correct percentages for class 0, class 1, and overall
                # Class 0 Correct (Accuracy of predicting 0 given true Y is 0)
                results['perCorr'][ch, 0] = 100 * nansum((tmp_test == 0) & (Y_test_01 == 0)) / np.sum(Y_test_01 == 0)
                # Class 1 Correct
                results['perCorr'][ch, 1] = 100 * nansum((tmp_test == 1) & (Y_test_01 == 1)) / np.sum(Y_test_01 == 1)
                # Overall Correct
                results['perCorr'][ch, 2] = 100 * nansum(tmp_test == Y_test_01) / len(test_ind)
                
                # Training Set Metrics
                results['perCorrTr'][ch, 0] = 100 * nansum((tmp_tr == 0) & (Y_tr_01 == 0)) / np.sum(Y_tr_01 == 0)
                results['perCorrTr'][ch, 1] = 100 * nansum((tmp_tr == 1) & (Y_tr_01 == 1)) / np.sum(Y_tr_01 == 1)
                results['perCorrTr'][ch, 2] = 100 * nansum(tmp_tr == Y_tr_01) / len(tr_ind)
            
            else:
                results['predChoice'][ch] = np.nan
                results['predChoiceTr'][ch] = np.nan
                results['perCorr'][ch, :] = np.nan
                results['perCorrTr'][ch, :] = np.nan
                
            results['et'][ch] = time_now() - t_start

    elif method == 'SVMR':
        results['predY'] = {}
        results['Y'] = {}
        results['gofTest'] = np.zeros(len(iChunk) - 1) * np.nan
        results['gofTr'] = np.zeros(len(iChunk) - 1) * np.nan
        
        for ch in range(len(iChunk) - 1):
            test_ind, tr_ind = get_indices(ch)
            
            if len(np.unique(Y[tr_ind])) > 1:
                # MATLAB: mY = 0; svmModel = fitrsvm(X(trInd,:), Y(trInd)-mY, ...);
                mY = 0.0 # Assumed mean subtraction is 0, as in the MATLAB code
                Y_tr_centered = Y[tr_ind] - mY
                
                # Fit SVR model
                # MATLAB 'KernelScale' corresponds to 'gamma' or scale factor
                # fitrsvm is Support Vector Regression (SVR)
                svmModel = SVR(kernel='rbf', gamma=1.0 / (2 * extras.get('kernelScale', 1.0)**2)) 
                svmModel.fit(X[tr_ind, :], Y_tr_centered)
                
                # Predict
                results['predY'][ch] = svmModel.predict(X[test_ind, :]) + mY
                results['Y'][ch] = Y[test_ind]
                
                predY_tr = svmModel.predict(X[tr_ind, :]) + mY
                
                # Calculate Goodness of Fit (Correlation)
                results['gofTest'][ch] = corrcoeff(results['predY'][ch], Y[test_ind])
                results['gofTr'][ch] = corrcoeff(predY_tr, Y[tr_ind])

    elif method == 'GLM':
        # Assuming Normal (Gaussian) GLM, equivalent to OLS (Ordinary Least Squares)
        results['W'] = np.zeros((X.shape[1] + 1, len(iChunk) - 1)) * np.nan # Weights + Intercept
        results['predY'] = {}
        results['rTr'] = np.zeros(len(iChunk) - 1) * np.nan
        results['rTest'] = np.zeros(len(iChunk) - 1) * np.nan
        
        # We need a GLM implementation (using a placeholder based on OLS/statsmodels GLM)
        from statsmodels.api import GLM, families, add_constant
        
        for ch in range(len(iChunk) - 1):
            test_ind, tr_ind = get_indices(ch)
            
            # MATLAB: glmfit(X(trInd,:),Y(trInd),'normal') -> OLS with intercept
            X_tr_const = add_constant(X[tr_ind, :], prepend=True)
            X_test_const = add_constant(X[test_ind, :], prepend=True)
            
            # Fit GLM (Normal family, Identity link) -> OLS
            glm_model = GLM(Y[tr_ind], X_tr_const, family=families.Gaussian())
            glm_results = glm_model.fit()
            
            # Store weights (W) and predictions (predY)
            # MATLAB: results.W(:,ch) = glmfit(...)
            results['W'][:, ch] = glm_results.params
            
            # MATLAB: results.predY{ch} = glmval(results.W(:,ch),X(testInd,:),'identity');
            results['predY'][ch] = glm_results.predict(X_test_const)
            predY_tr = glm_results.predict(X_tr_const)
            
            # Correlation (rTr and rTest)
            results['rTr'][ch] = corrcoeff(predY_tr, Y[tr_ind])
            results['rTest'][ch] = corrcoeff(results['predY'][ch], Y[test_ind])

    elif method == 'BLR': # Binomial Logistic Regression
        # MATLAB uses mnrfit/mnrval (multinomial logistic regression) but with binary Y.
        # Equivalent to standard Logistic Regression for binary classification.
        results['aucTr'] = np.zeros(len(iChunk) - 1) * np.nan
        results['aucTest'] = np.zeros(len(iChunk) - 1) * np.nan
        results['et'] = np.zeros(len(iChunk) - 1)
        results['B'] = np.zeros((X.shape[1] + 1, len(iChunk) - 1)) * np.nan
        results['classProbTr'] = {}
        results['classProbTest'] = {}
        
        # Using statsmodels GLM or Scikit-learn LogisticRegression
        from sklearn.linear_model import LogisticRegression
        
        for ch in range(len(iChunk) - 1):
            t_start = time_now()
            test_ind, tr_ind = get_indices(ch)
            results['trInd'] = tr_ind # Note: This only stores the indices of the last run
            results['testInd'] = test_ind # This only stores the indices of the last run
            
            # Check for binary classification
            if len(np.unique(Y[tr_ind])) > 1:
                # Convert MATLAB labels (1 and 2) to Python standard (0 and 1) if necessary
                Y_tr_01 = Y[tr_ind] - 1 if np.max(Y) > 1 else Y[tr_ind]
                Y_test_01 = Y[test_ind] - 1 if np.max(Y) > 1 else Y[test_ind]
                
                # MATLAB: results.B(:,ch) = mnrfit(X(trInd,:), Y(trInd));
                # Scikit-learn LogisticRegression implicitly handles the intercept
                log_reg = LogisticRegression(penalty=None, solver='lbfgs', fit_intercept=True)
                log_reg.fit(X[tr_ind, :], Y_tr_01)
                
                # Store weights (intercept first, then coefficients)
                # MATLAB mnrfit output is (Intercept, Coeff1, Coeff2, ...)
                results['B'][:, ch] = np.concatenate(([log_reg.intercept_[0]], log_reg.coef_[0]))
                
                # MATLAB: results.classProbTr/Test{ch} = mnrval(...) -> Probabilities
                # Scikit-learn predict_proba returns [P(Y=0), P(Y=1)]
                prob_tr = log_reg.predict_proba(X[tr_ind, :])
                prob_test = log_reg.predict_proba(X[test_ind, :])
                
                # MATLAB mnrval for binary Y (1, 2) often returns a 2-column matrix 
                # where the columns correspond to the classes (or similar to predict_proba).
                # We store the [P(Y=0), P(Y=1)] equivalent.
                results['classProbTr'][ch] = prob_tr
                results['classProbTest'][ch] = prob_test
                
                # AUC calculation (perfcurve equivalent)
                # We need the probability of the positive class (Y=1, which is column 2)
                
                # Training AUC
                # Probabilities for positive class (column index 1 in 0-based indexing)
                # MATLAB: Y(trInd) == 1 (Negative class), Y(trInd) == 2 (Positive class)
                # MATLAB AUC logic is based on: [zeros(1,len(a)), ones(1,len(b))], [a b], 1
                # where 'a' are scores for negative class, 'b' are scores for positive class.
                
                # In our 0/1 setup: Y=0 is negative, Y=1 is positive.
                if np.sum(Y_tr_01 == 0) > 0 and np.sum(Y_tr_01 == 1) > 0:
                    results['aucTr'][ch] = roc_auc_score(Y_tr_01, prob_tr[:, 1])

                # Testing AUC
                if len(np.unique(Y_test_01)) > 1:
                    if np.sum(Y_test_01 == 0) > 0 and np.sum(Y_test_01 == 1) > 0:
                        try:
                            results['aucTest'][ch] = roc_auc_score(Y_test_01, prob_test[:, 1])
                        except ValueError:
                            # Handle errors if prob_test or Y_test_01 contains problematic values
                            warnings.warn(f"AUC calculation failed in test set for chunk {ch}. Skipping plot.")
                            pass
                            
            results['et'][ch] = time_now() - t_start

    elif method == 'CCA':
        # Placeholder for CCA (Canonical Correlation Analysis)
        # Using Scikit-learn's CCA as a rough approximation, ignoring PMDmac/kernel scale/filtering details
        from sklearn.cross_decomposition import CCA
        
        fs = extras.get('fs', [0.1])
        nComp = extras.get('nComp', 1)
        nChunk = len(iChunk)
        
        results = [{} for _ in fs] # results is a cell array indexed by fs (f)
        
        for f, fs_val in enumerate(fs):
            results[f]['A'] = np.zeros((X.shape[1], nComp, nChunk - 1)) * np.nan
            results[f]['B'] = np.zeros((Y.shape[1], nComp, nChunk - 1)) * np.nan
            results[f]['r_tr'] = np.zeros((nComp, nChunk - 1)) * np.nan
            results[f]['r_test'] = np.zeros((nComp, nChunk - 1)) * np.nan
            results[f]['et'] = np.zeros(nChunk - 1)
            
            for ch in range(nChunk - 1):
                t_start = time_now()
                test_ind, tr_ind = get_indices(ch)
                
                # MATLAB filtering NaNs in training data (iValid)
                # Assuming Y is already 2D (Y(trInd,1)) check
                tr_data = np.hstack([X[tr_ind, :], Y[tr_ind, :]])
                iInvalid_mask = ~np.isfinite(np.sum(tr_data, axis=1))
                iValid = tr_ind[~iInvalid_mask] # Valid indices for training
                
                if len(iValid) > 0:
                    # MATLAB: [A,B,r_tr] = canoncorrPMDmac(X(iValid,:),Y(iValid,:), nComp, fs(f), 1);
                    
                    # Placeholder CCA model (ignoring PMD and fs_val)
                    try:
                        cca_model = CCA(n_components=nComp)
                        cca_model.fit(X[iValid, :], Y[iValid, :])
                        
                        # A = X weights, B = Y weights
                        A_cca = cca_model.x_weights_
                        B_cca = cca_model.y_weights_
                        
                        # Calculate training correlation (r_tr)
                        r_tr = np.array([corrcoeff(X[iValid, :] @ A_cca[:, c], Y[iValid, :] @ B_cca[:, c]) 
                                        for c in range(nComp)])
                        
                        # Sort by r_tr (ascending in MATLAB)
                        ind = np.argsort(r_tr)
                        
                        results[f]['r_tr'][:, ch] = r_tr[ind]
                        results[f]['A'][:, :, ch] = A_cca[:, ind]
                        results[f]['B'][:, :, ch] = B_cca[:, ind]
                        
                        # Test correlation (r_test)
                        for c_idx in range(nComp):
                            # Use sorted weights for test correlation
                            A_comp = results[f]['A'][:, c_idx, ch]
                            B_comp = results[f]['B'][:, c_idx, ch]
                            
                            r_test = corrcoeff(X[test_ind, :] @ A_comp, Y[test_ind, :] @ B_comp)
                            results[f]['r_test'][c_idx, ch] = r_test
                            
                    except ValueError:
                        warnings.warn(f"CCA failed for chunk {ch} (f={f}). Check data size/variance.")
                        pass # CCA failed
                        
                results[f]['et'][ch] = time_now() - t_start

    elif method == 'pCorr':
        results['mxCorr'] = np.zeros((len(iChunk) - 1, X.shape[1])) * np.nan
        nChunk = len(iChunk)
        
        for ch in range(nChunk - 1):
            test_ind, tr_ind = get_indices(ch)
            
            # Mean and Std (Training Data)
            mX = nanmean(X[tr_ind, :], axis=0)
            sX = np.nanstd(X[tr_ind, :], axis=0, ddof=1) # ddof=1 for sample std
            mY = nanmean(Y[tr_ind, :], axis=0)
            sY = np.nanstd(Y[tr_ind, :], axis=0, ddof=1)
            
            # MATLAB: xyCorr (Pearson Correlation on Training Set)
            # This calculation is for the covariance normalized by stds (Pearson r)
            xyCorr = np.zeros((X.shape[1], Y.shape[1]))
            
            for j1 in range(X.shape[1]):
                for j2 in range(Y.shape[1]):
                    # Calculate the numerator (covariance) using nanmean on the product
                    cov_num = nanmean((X[tr_ind, j1] - mX[j1]) * (Y[tr_ind, j2] - mY[j2]))
                    
                    # Normalize by stds
                    if sX[j1] > 0 and sY[j2] > 0:
                        xyCorr[j1, j2] = cov_num / (sX[j1] * sY[j2])
                    else:
                        xyCorr[j1, j2] = np.nan
            
            # Find max correlation for each X variable (X.shape[1])
            for j in range(X.shape[1]):
                absFlag = extras.get('absFlag', 0)
                
                if absFlag == 1:
                    # Max of absolute correlation
                    mxCorrTr = np.max(np.abs(xyCorr[j, :]))
                    mxInd = np.argmax(np.abs(xyCorr[j, :]))
                    sg = np.sign(xyCorr[j, mxInd])
                else:
                    # Max correlation (signed)
                    mxCorrTr = np.max(xyCorr[j, :])
                    mxInd = np.argmax(xyCorr[j, :])
                    sg = 1 # Sign is already in mxCorrTr
                
                # Test Correlation (using max-correlated Y-variable index mxInd)
                # MATLAB: mxCorr(ch,j) = sg * nanmean((X(testInd,j) - mX(j)) .* (Y(testInd,mxInd) - mY(mxInd))./(sX(j).*sY(mxInd)));
                
                cov_num_test = nanmean((X[test_ind, j] - mX[j]) * (Y[test_ind, mxInd] - mY[mxInd]))
                
                if sX[j] > 0 and sY[mxInd] > 0:
                    r_test = cov_num_test / (sX[j] * sY[mxInd])
                else:
                    r_test = np.nan
                    
                results['mxCorr'][ch, j] = sg * r_test
                
    else:
        raise ValueError(f"Unknown method: {method}")

    return results