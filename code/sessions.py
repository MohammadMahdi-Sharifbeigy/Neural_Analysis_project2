import numpy as np
import scipy.io as sio
import os
import matplotlib.pyplot as plt
from pathlib import Path
import warnings

# --- Utility Functions (Mimicking MATLAB) ---

def nanmean(data):
    """Simple wrapper for nanmean."""
    return np.nanmean(data)

def isempty(arr):
    """Mimics MATLAB's isempty for list/array."""
    return len(arr) == 0

def prctile(data, q):
    """Mimics MATLAB's prctile (percentiles/quantiles)."""
    # np.percentile handles NaNs by default if asked, but let's filter just in case
    data = np.array(data)
    data = data[np.isfinite(data)]
    if data.size == 0:
        return np.full_like(q, np.nan)
    return np.percentile(data, q)

def discretize(data, bins):
    """Mimics MATLAB's discretize using np.digitize."""
    # np.digitize returns the index of the *right* bin edge
    # MATLAB's discretize behavior is usually 1-based, so we adjust.
    return np.digitize(data, bins)

# --- Define Paths and Data Loading Function ---

# Determine project and code paths
CODE_PATH = Path(os.getcwd())
# MATLAB: projPath = codePath(1:end-4); -> assumes project root is 4 levels up
PROJ_PATH = CODE_PATH.parent.parent.parent.parent

# Mock function for loading .mat files
def load_mat_data(file_path):
    """Attempts to load a .mat file using scipy.io.loadmat."""
    full_path = PROJ_PATH / file_path
    if not full_path.exists():
        warnings.warn(f"File not found: {full_path}. Returning empty dictionary.")
        return {}
    
    # MATLAB arrays often come in as (N, 1) or (1, N) in Python, 
    # and strings/cell arrays are nested lists/arrays.
    try:
        return sio.loadmat(full_path, squeeze_me=True)
    except Exception as e:
        warnings.warn(f"Error loading {full_path}: {e}")
        return {}

# --- 1. Session Setup ---

# MATLAB: clear all (Equivalent to starting with a clean Python script state)
i = 1
sessionNames = {}
sch = {}
schRev = {}

# The following structure is repeated for many sessions:
def add_session(name, schedule, is_reversed=0):
    global i
    sessionNames[i] = name
    sch[i] = schedule
    if is_reversed == 1:
        schRev[i] = 1 # Only set if reversed
    i += 1

# --- Gin Sessions ---
add_session('Gin_03Jun15_block1', [20, 20, 25])
add_session('Gin_03Jun15_block2', [20, 20, 17])
add_session('Gin_04Jun15_block1', [20, 20, 55])
add_session('Gin_04Jun15_block2', [20, 20, 46])
add_session('Gin_05Jun15_block1', [30, 30, 38])
add_session('Gin_05Jun15_block3', [30, 30, 40])
add_session('Gin_10Jun15_block1', [30, 30, 29])
add_session('Gin_12Jun15_block1', [np.nan, np.nan, 25])
add_session('Gin_15Jun15_block1', [np.nan, np.nan, 29])
add_session('Gin_18Jun15_block2', [20, 20, 85])
add_session('Gin_22Jun15_block1', [25, 25, 97])
add_session('Gin_24Jun15_block1', [25, 25, 98])
add_session('Gin_26Jun15_block1', [30, 30, 98])
add_session('Gin_29Jun15_block1', [20, 40, 107])
add_session('Gin_01Jul15_block1', [20, 40, 93])
add_session('Gin_03Jul15_block1', [20, 40, 50])
add_session('Gin_03Jul15_block2', [20, 40, 50])
add_session('Gin_07Jul15_block1', [20, 40, 92])
add_session('Gin_09Jul15_block1', [20, 40, 99])
add_session('Gin_13Jul15_block1', [20, 40, 100])
add_session('Gin_27Jul15_block1', [10, 40, 98])

# --- Tony Sessions (with comments indicating complexity/mismatch) ---
add_session('Tony_30Jul15_block1', [10, 10, 137])
add_session('Tony_04Aug15_block1', [10, 10, 33, 15, 15, 76, 15, 15, 25]) # mismatch
add_session('Tony_07Aug15_block1', [20, 20, 84, 10, 10, 78]) # mismatch
add_session('Tony_10Aug15_block1', [10, 10, 150]) # mismatch
add_session('Tony_12Aug15_block1', [10, 10, 135])
add_session('Tony_13Aug15_block1', [10, 10, 144])
add_session('Tony_18Aug15_block1', [25, 15, 110])
add_session('Tony_19Aug15_block1', [25, 15, 92])

# --- Gin Sessions (Reversed Schedule) ---
add_session('Gin_31Aug15_block1', [25, 15, 34, 15, 25, 66], 1)
add_session('Gin_01Sep15_block1', [15, 25, 34, 25, 15, 66], 1)
add_session('Gin_02Sep15_block1', [25, 15, 34, 15, 25, 66], 1)
add_session('Gin_03Sep15_block1', [15, 25, 34, 25, 15, 66], 1)
add_session('Gin_04Sep15_block1', [25, 15, 34, 15, 25, 66], 1)
add_session('Gin_08Sep15_block1', [25, 15, 34, 15, 25, 66], 1)
add_session('Gin_09Sep15_block1', [15, 25, 34, 25, 15, 66], 1)
add_session('Gin_11Sep15_block1', [15, 25, 34, 25, 15, 66], 1)
add_session('Gin_14Sep15_block1', [15, 25, 34, 25, 15, 66], 1)
add_session('Gin_16Sep15_block1', [25, 15, 34, 15, 25, 66], 1)
add_session('Gin_17Sep15_block1', [25, 15, 34, 15, 25, 66], 1)
add_session('Gin_21Sep15_block1', [25, 15, 34, 15, 25, 66], 1)
add_session('Gin_22Sep15_block1', [15, 25, 34, 25, 15, 66], 1)
add_session('Gin_28Sep15_block1', [15, 25, 34, 25, 15, 66], 1)
add_session('Gin_29Sep15_block1', [20, 20, 98])
add_session('Gin_30Sep15_block1', [20, 20, 100])
add_session('Gin_06Oct15_block1', [15, 25, 34, 25, 15, 66], 1)

# --- Tony Sessions (Continued) ---
add_session('Tony_29Sep15_block1', [20, 20, 50, 20, 20, 50])
add_session('Tony_06Oct15_block1', [30, 30, 91])
add_session('Tony_22Oct15_block1', [25, 25, 99])
add_session('Tony_29Oct15_block1', [np.nan, np.nan, 0])
add_session('Tony_04Nov15_block1', [30, 30, 50])
add_session('Tony_04Nov15_block3', [30, 30, 46])
add_session('Tony_05Nov15_block1', [20, 20, 46])
add_session('Tony_05Nov15_block3', [20, 20, 47])
add_session('Tony_06Nov15_block1', [30, 30, 47])
add_session('Tony_06Nov15_block4', [30, 30, 40])
add_session('Tony_09Nov15_block1', [30, 30, 50])
add_session('Tony_09Nov15_block4', [30, 30, 50])
add_session('Tony_12Nov15_block1', [30, 30, 69])
add_session('Tony_12Nov15_block3', [30, 30, 69])
add_session('Tony_01Dec15_block1', [20, 20, 99])
add_session('Tony_01Dec15_block3', [20, 20, 96])
add_session('Tony_02Dec15_block1', [25, 15, 34, 15, 25, 66], 1)
add_session('Tony_02Dec15_block3', [25, 15, 34, 15, 25, 66], 1)
add_session('Tony_03Dec15_block1', [20, 40, 100])
add_session('Tony_03Dec15_block3', [20, 40, 100])
add_session('Tony_04Dec15_block1', [25, 15, 34, 15, 25, 66], 1)
add_session('Tony_04Dec15_block3', [15, 25, 34, 25, 15, 66], 1)
add_session('Tony_07Dec15_block1', [30, 30, 100, 30, 30, 100])
add_session('Tony_08Dec15_block1', [15, 25, 34, 25, 15, 66, 15, 25, 34, 25, 15, 66])
add_session('Tony_09Dec15_block1', [20, 40, 202])
add_session('Tony_10Dec15_block1', [20, 40, 194])
add_session('Tony_11Dec15_block1', [25, 15, 34, 15, 25, 66, 25, 15, 34], 1)
add_session('Tony_14Dec15_block1', [25, 15, 34, 15, 25, 66, 15, 25, 34, 25, 15, 66], 1)
add_session('Tony_15Dec15_block1', [15, 25, 34, 25, 15, 66, 20, 20, 100], 1)
add_session('Tony_16Dec15_block1', [15, 25, 34, 25, 15, 66, 15, 25, 34, 25, 15, 66], 1)
add_session('Tony_17Dec15_block1', [30, 15, 199])
add_session('Tony_18Dec15_block1', [25, 15, 34, 15, 25, 66, 15, 25, 34, 25, 15, 66], 1)
add_session('Tony_21Dec15_block1', [15, 25, 34, 25, 15, 66, 15, 25, 34, 25, 15, 66], 1)
add_session('Tony_22Dec15_block1', [25, 15, 34, 15, 25, 66, 15, 25, 34, 25, 15, 66], 1)
add_session('Tony_23Dec15_block1', [30, 15, 197])
add_session('Tony_24Dec15_block1', [30, 15, 199])
add_session('Tony_27Dec15_block1', [15, 25, 34, 25, 15, 66, 15, 25, 34, 25, 15, 66], 1)
add_session('Tony_28Dec15_block1', [25, 15, 34, 15, 25, 66, 15, 25, 34, 25, 15, 66], 1)
add_session('Tony_29Dec15_block1', [10, 30, 34, 30, 10, 66, 10, 30, 34, 30, 10, 66], 1)
add_session('Tony_30Dec15_block1', [10, 30, 34, 30, 10, 66, 10, 30, 34, 30, 10, 66], 1)
add_session('Tony_31Dec15_block1', [30, 10, 34, 10, 30, 66, 10, 30, 34, 30, 10, 66], 1)
add_session('Tony_18Jan16_block1', [10, 30, 201])
add_session('Tony_19Jan16_block1', [10, 30, 197])
add_session('Tony_20Jan16_block1', [10, 30, 192])
add_session('Tony_21Jan16_block1', [5, 20, 100, 5, 20, 100])
add_session('Tony_22Jan16_block1', [5, 20, 100, 10, 30, 94])
add_session('Tony_25Jan16_block1', [10, 30, 25, 5, 30, 155])
add_session('Tony_27Jan16_block1', [15, 25, 100, 15, 25, 34, 25, 15, 66], 1)
add_session('Tony_29Jan16_block1', [10, 30, 152])
add_session('Tony_01Feb16_block1', [10, 30, 100, 10, 30, 34, 30, 10, 66], 1)
add_session('Tony_02Feb16_block1', [30, 10, 34, 10, 30, 66, 10, 30, 34, 30, 10, 66], 1)
add_session('Tony_03Feb16_block1', [30, 10, 34, 10, 30, 66, 10, 30, 100], 1)
add_session('Tony_04Feb16_block1', [10, 30, 34, 30, 10, 66, 30, 10, 34, 10, 30, 66], 1)
add_session('Tony_08Feb16_block1', [25, 15, 34, 15, 25, 66, 15, 25, 34, 25, 15, 66], 1)
add_session('Tony_09Feb16_block1', [25, 15, 100, 25, 15, 100]) # No schRev, mismatch comment
add_session('Tony_10Feb16_block1', [15, 25, 34, 25, 15, 66, 15, 25, 100], 1) # mismatch comment
add_session('Tony_17Feb16_block1', [25, 15, 34, 15, 25, 66], 1) # mismatch comment
add_session('Tony_19Feb16_block1', [15, 25, 34, 25, 15, 66], 1) # mismatch comment
add_session('Tony_24Feb16_block1', [15, 25, 34, 25, 15, 66, 25, 15, 34, 15, 25, 66], 1) # mismatch comment
add_session('Tony_26Feb16_block1', [25, 15, 34, 15, 25, 66, 25, 15, 34, 15, 25, 66], 1)
add_session('Tony_29Feb16_block1', [25, 15, 34, 15, 25, 66, 25, 15, 34, 15, 25, 66], 1) # mismatch comment
add_session('Tony_02Mar16_block1', [15, 25, 34, 25, 15, 66, 25, 15, 34, 15, 25, 66], 1)
add_session('Tony_04Mar16_block1', [25, 15, 34, 15, 25, 66, 25, 15, 34, 15, 25, 66], 1)
add_session('Tony_07Mar16_block1', [15, 25, 34, 25, 15, 66, 25, 15, 34, 15, 25, 66], 1)
add_session('Tony_09Mar16_block1', [15, 25, 34, 25, 15, 66, 25, 15, 34, 15, 25, 66], 1)
add_session('Tony_11Mar16_block1', [15, 25, 34, 25, 15, 66, 25, 15, 34, 15, 25, 66], 1)

# --- 2. Post-Setup Variables ---

numSessions = len(sessionNames)
# MATLAB: codePath = pwd; projPath = codePath(1:end-4);
# Already handled by PROJ_PATH and CODE_PATH at the top

# MATLAB: addpath(genpath([projPath '/codingLib']))
# This is a MATLAB-specific environment setup and is skipped in Python.

numChan = 96

# Define session index lists
# Note: MATLAB indices (1-based) are used in the original script
GinSessions = np.array([10, 11, 12, 13, 14, 16, 17, 18, 20, 21, 31, 33, 34, 35, 36, 38, 39, 40, 41, 42, 43, 44, 45, 46])
TonySessions = np.array([22, 24, 26, 27, 28, 29, 47, 48, 49, 51, 52, 53, 54, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 92, 95, 97, 98, 99, 102, 106, 108, 109, 110, 111, 112])
GinCutOff = 46 # Index, likely used for splitting Gin sessions based on experimental phase

# --- 3. Load Combined Behavioral Data ---

behavValidVer = '2p7'
# MATLAB: load([projPath 'data/sessionsCombined/behav_valid' behavValidVer '.mat' ]);
behav_data = load_mat_data(f'data/sessionsCombined/behav_valid{behavValidVer}.mat')

# Extract key variables (assuming successful load)
# These will be empty or placeholders if the file was not found
tslpAll = behav_data.get('tslpAll', np.array([]))
kslwAll = behav_data.get('kslwAll', np.array([]))
rewRateAll = behav_data.get('rewRateAll', np.array([]))
rewRatioAll = behav_data.get('rewRatioAll', np.array([]))
tslwAll = behav_data.get('tslwAll', np.array([]))
schAll = behav_data.get('schAll', np.array([]))
pRewAll = behav_data.get('pRewAll', np.array([]))
choiceAll = behav_data.get('choiceAll', np.array([]))
sessionsAll = behav_data.get('sessionsAll', np.array([]))
tunpAll = behav_data.get('tunpAll', np.array([]))
b1PushedTimes = behav_data.get('b1PushedTimes', {}) # Assuming cell array -> dictionary/list of arrays
bLocX = behav_data.get('bLocX', {}) # Assuming cell array -> dictionary/list of arrays
pRew = behav_data.get('pRew', {}) # Assuming cell array -> dictionary/list of arrays
tslp = behav_data.get('tslp', {}) # Assuming cell array -> dictionary/list of arrays
iForag = behav_data.get('iForag', {}) # Assuming cell array -> dictionary/list of arrays


validSessions = behav_data.get('validSessions', np.array([
    31, 32, 33, 34, 38, 39, 40, 41, 42, 45, 46, 63, 64, 67, 68, 69, 71, 74, 76, 79, 
    80, 84, 95, 97, 98, 99, 109, 110, 111, 112
]))

# --- 4. Feature Engineering and Predictors ---

nClasses = 4

# MATLAB: actionAll = nClasses*choiceAll; % switch is the last class
# Assume choiceAll contains values that discretize 'action'
actionAll = nClasses * choiceAll 

# MATLAB: irtBins = prctile(tslpAll, [linspace(0,100,nClasses)]);
irtBins = prctile(tslpAll, np.linspace(0, 100, nClasses))

# MATLAB: tmp = tunpAll(choiceAll==0);
tmp = tunpAll[choiceAll == 0]

# MATLAB: actionAll(choiceAll==0) = discretize(tmp, irtBins);
# Discretize tmp (where choiceAll=0) and assign back to actionAll
if tmp.size > 0:
    # discretize returns 1-based index (1, 2, 3, 4 for nClasses=4)
    discrete_labels = discretize(tmp, irtBins) 
    actionAll[choiceAll == 0] = discrete_labels

# MATLAB: rewPredAll = [(log(tslpAll))' 1./(kslwAll)' rewRateAll' rewRatioAll' 1./(tslwAll)' 1./schAll' pRewAll'];
# Create predictor matrix (ensure all inputs are 1D arrays or column vectors)
# Use np.column_stack to create the final matrix
with np.errstate(divide='ignore', invalid='ignore'):
    rewPredAll = np.column_stack([
        np.log(tslpAll.T) if tslpAll.ndim > 1 else np.log(tslpAll),
        1.0 / (kslwAll.T) if kslwAll.ndim > 1 else 1.0 / kslwAll,
        rewRateAll.T if rewRateAll.ndim > 1 else rewRateAll,
        rewRatioAll.T if rewRatioAll.ndim > 1 else rewRatioAll,
        1.0 / (tslwAll.T) if tslwAll.ndim > 1 else 1.0 / tslwAll,
        1.0 / (schAll.T) if schAll.ndim > 1 else 1.0 / schAll,
        pRewAll.T if pRewAll.ndim > 1 else pRewAll
    ])

rewPredNameAll = ['log(tslp)', '1/(losses+1)', 'rew/resp', 'lrr', 'ins rew rate', '1/sch', 'pRew']
nRewPreds = len(rewPredNameAll)

# MATLAB: prevChoiceAll = [nan choiceAll(1:end-1)];
if choiceAll.size > 0:
    prevChoiceAll = np.concatenate(([np.nan], choiceAll[:-1]))
else:
    prevChoiceAll = np.array([])


# MATLAB: XbaseAll = [rewAll' prevChoiceAll' sessionsAll' ];
# NOTE: 'rewAll' is not defined in the provided script snippet. Assuming it exists.
# Using a placeholder for rewAll, then combining.
rewAll_placeholder = np.zeros_like(choiceAll) if choiceAll.size > 0 else np.array([])
XbaseAll = np.column_stack([
    rewAll_placeholder.T if rewAll_placeholder.ndim > 1 else rewAll_placeholder,
    prevChoiceAll.T if prevChoiceAll.ndim > 1 else prevChoiceAll,
    sessionsAll.T if sessionsAll.ndim > 1 else sessionsAll
])

# --- 5. Session Filtering and Data Combination ---

# Identify sessions with behavioral data (b1PushedTimes is not empty)
behavSessions = []
# MATLAB: for s = 1:numSessions, if (~isempty(b1PushedTimes{s})) ...
for s_idx in range(1, numSessions + 1):
    # Check if the session index exists in the dictionary and its content is not empty
    if s_idx in b1PushedTimes and not isempty(b1PushedTimes.get(s_idx, [])):
        behavSessions.append(s_idx)

# Identify sessions with localization data (bLocX is not empty)
locSessions = []
for s_idx in range(1, numSessions + 1):
    if s_idx in bLocX and not isempty(bLocX.get(s_idx, [])):
        locSessions.append(s_idx)

binSize = 200

# MATLAB: load([codePath '/../data/sessionsCombined/decorMatices.mat']);
decor_data = load_mat_data('data/sessionsCombined/decorMatices.mat')
# decor_data will contain variables like decorMatices if loaded successfully

pRewAll0 = []
pRewAll_filtered = []

# Loop over validSessions to combine and filter pRew data
for s in validSessions:
    # Assuming tslp and iForag are dictionaries/lists indexed by session number (s)
    if s in tslp and s in iForag and s in pRew:
        # MATLAB: ind = find(tslp{s}(iForag{s})<=60 & tslp{s}(iForag{s})>2);
        
        # Get data for foraging trials
        tslp_s_forag = tslp[s][iForag[s]]
        pRew_s_forag = pRew[s][iForag[s]]
        
        # Filter indices: tslp between 2 and 60 (inclusive of 2, exclusive of 60)
        # Note: MATLAB's find and indexing return 1-based indices, Python uses boolean masks
        ind_mask = (tslp_s_forag <= 60) & (tslp_s_forag > 2)
        
        # pRewAll0 = [pRewAll0 pRew{s}(iForag{s})]; (All foraging rewards)
        pRewAll0.extend(pRew_s_forag)
        
        # pRewAll = [pRewAll pRew{s}(iForag{s}(ind))]; (Filtered foraging rewards)
        pRewAll_filtered.extend(pRew_s_forag[ind_mask])

# Convert lists to NumPy arrays
pRewAll0 = np.array(pRewAll0)
pRewAll_filtered = np.array(pRewAll_filtered)

# --- 6. Plotting and Color Setup ---

# MATLAB: clf, histogram(pRewAll0, 0:0.05:1), hold on, histogram(pRewAll, 0:0.05:1)
plt.figure()
bins = np.arange(0, 1.05, 0.05)
plt.hist(pRewAll0, bins=bins, alpha=0.6, label='pRewAll0 (All Foraging)')
plt.hist(pRewAll_filtered, bins=bins, alpha=0.6, label='pRewAll (Filtered Foraging)')
plt.xlabel('Probability of Reward (pRew)')
plt.ylabel('Count')
plt.title('Histogram of Reward Probability')
plt.legend()
plt.show(block=False)

# Color and Jitter setup
# MATLAB: jit = 0.05*randn(1,100000);
jit = 0.05 * np.random.randn(100000)

gray = [0.5, 0.5, 0.5]
tslpCl = [0, 0.6, 0]
lrrCl = [0, 0, 1]
choiceCl = [1, 0.5, 0]
rewCl = [1, 0, 0]

codeVer = '_shared'

# The script finishes here.