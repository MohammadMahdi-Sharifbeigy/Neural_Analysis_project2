import numpy as np
from scipy.stats import norm, rankdata
from scipy.special import logit, expit
from .roc import roc 

def tiedrank(data):
    """
    MATLAB: R = tiedrank(X)
    Uses scipy.stats.rankdata with 'average' method, which is the default 
    behavior for tiedrank in MATLAB for computing ranks (1-based).
    """
    # ranks are 1-based (MATLAB default)
    return rankdata(data, method='average')

def auc(data, alpha=0.05, flag='logit', nboot=1000, **kwargs):
    """
    MATLAB: function [A,Aci] = auc(data,alpha,flag,nboot,varargin)
    
    Calculates the Area Under the ROC Curve (AUC) and confidence intervals.
    
    Returns:
        tuple: (A, Aci) - Area under ROC (A) and confidence intervals (Aci or None).
    """
    
    # Determine if Aci is requested (simulate nargout == 2)
    # If two output variables are not received in Python, Aci should be None.
    # Here, we always calculate CI unless explicitly turned off.
    # (For ease of reuse by auc_bootstrap, this function only returns AUC
    # if nargout == 1, which in Python is not distinguishable from the number of requested outputs
    # so we act based on the needs of the calling function.)
    calculate_Aci = True # Always calculate CI unless nboot is called

    if data.shape[1] != 2:
       raise ValueError('Incorrect input size in AUC!')

    flag = flag.lower()
    
    # --- ورودی‌های اختیاری (شبیه‌سازی default) ---
    if alpha is None: alpha = 0.05
    if flag is None or flag == '': flag = 'logit'
    if nboot is None: nboot = 1000

    # Count observations by class
    m = np.sum(data[:, 0] > 0) # Positive class count
    n = np.sum(data[:, 0] <= 0) # Negative class count

    # Handle case with only one class
    if m == 0 or n == 0:
        return 0.5, np.array([np.nan, np.nan]) if calculate_Aci else 0.5

    # --- محاسبه AUC ---
    # [tp,fp] = roc(data);
    # توجه: باید تابع roc در دسترس باشد.
    try:
        # فرض می‌کنیم roc در scope موجود است (در پایتون، باید import شود)
        tp, fp = roc(data) 
    except NameError:
        # در صورت نبود تابع roc، از np.trapz استفاده می‌کنیم که منطق AUC را محاسبه می‌کند
        # اما نیاز به محاسبه ROC به صورت دستی دارد (که در roc.m انجام شده است).
        # برای دقت، ما فرض می‌کنیم roc در دسترس است و در غیر این صورت خطا می‌دهیم.
        # اما برای حل مسئله، از روش تراپز داخلی MATLAB استفاده می‌کنیم.
        # MATLAB: A = sum((fp(2:end) - fp(1:end-1)).*(tp(2:end) + tp(1:end-1)))/2;
        # np.trapz(y, x)
        
        # --- روش جایگزین بر اساس Rank-Sum (دقیق‌تر) ---
        # این روش در MATLAB به عنوان یک کامنت آمده اما اغلب دقیق‌تر است.
        # R = tiedrank(data(:,2));
        # A = (sum(R(data(:,1)==1)) - (m^2 + m)/2) / (m * n);
        try:
            R = tiedrank(data[:, 1])
            A = (np.sum(R[data[:, 0] > 0]) - (m**2 + m) / 2) / (m * n)
            A = np.clip(A, 0.0, 1.0) # محدود کردن AUC به [0, 1]
            
            # برای حفظ سازگاری با roc.m، ما باید roc را فراخوانی کنیم
            # و از روش trapezoidal (ذوزنقه‌ای) استفاده کنیم.
            tp, fp = roc(data)
            A = np.trapz(tp, fp)
            
        except NameError:
            raise NameError("The 'roc' and 'tiedrank' functions are required but not found in scope.")
    
    Aci = None
    
    # --- (Confidence Intervals) ---
    if calculate_Aci:
        N = m + n
        # MATLAB: z = norminv(1-alpha/2);
        z = norm.ppf(1 - alpha / 2)
        
        # Max Variance Standard Error term
        mv = np.sqrt((A * (1 - A)) / (0.75 * N - 1))
        
        if flag == 'hanley':
            Q1 = A / (2 - A)
            Q2 = (2 * A**2) / (1 + A)
            
            Avar = A * (1 - A) + (m - 1) * (Q1 - A**2) + (n - 1) * (Q2 - A**2)
            Avar = Avar / (m * n)       
            Ase = np.sqrt(Avar)
            Aci = np.array([A - z * Ase, A + z * Ase])
            
        elif flag == 'maxvar':
            Avar = (A * (1 - A)) / min(m, n)       
            Ase = np.sqrt(Avar)
            Aci = np.array([A - z * Ase, A + z * Ase])
            
        elif flag in ['mann-whitney', 'logit']:
            # Reverse labels to keep notation like Qin & Hotilovac (MATLAB code)
            m_neg = np.sum(data[:, 0] <= 0)
            n_pos = np.sum(data[:, 0] > 0)
            X = data[data[:, 0] <= 0, 1] # Negative scores
            Y = data[data[:, 0] > 0, 1] # Positive scores
            
            # Concat and rank (MATLAB: temp = [sort(X);sort(Y)]; temp = tiedrank(temp);)
            temp = np.concatenate((X, Y))
            temp_rank = tiedrank(temp)
            
            R = temp_rank[:m_neg] # Ranks for negative class
            S = temp_rank[m_neg:] # Ranks for positive class
            Rbar = np.mean(R)
            Sbar = np.mean(S)
            
            # MATLAB: (R-(1:m)').^2 (Ranks minus 1-based index)
            # در پایتون: (R - (np.arange(m_neg) + 1))**2
            S102_sum_term = np.sum((R - (np.arange(m_neg) + 1))**2)
            S102 = (1 / ((m_neg - 1) * n_pos**2)) * (S102_sum_term - m_neg * (Rbar - (m_neg + 1) / 2)**2)
            
            # MATLAB: (S-(1:n)').^2
            S012_sum_term = np.sum((S - (np.arange(n_pos) + 1))**2)
            S012 = (1 / ((n_pos - 1) * m_neg**2)) * (S012_sum_term - n_pos * (Sbar - (n_pos + 1) / 2)**2)
            
            S2 = (m_neg * S012 + n_pos * S102) / (m_neg + n_pos)
            
            Avar = ((m_neg + n_pos) * S2) / (m_neg * n_pos)
            Ase = np.sqrt(Avar)
            
            if flag == 'logit':
                # Logit transform: log(A/(1-A))
                logitA = logit(A) 
                
                # CI bounds on the logit scale
                LL = logitA - z * (Ase) / (A * (1 - A))
                UL = logitA + z * (Ase) / (A * (1 - A))
                
                # Inverse logit transform: exp(L)/(1+exp(L))
                Aci = np.array([expit(LL), expit(UL)])
            else: # 'mann-whitney'
                Aci = np.array([A - z * Ase, A + z * Ase])
                
        elif flag == 'wald': 
            Aci = np.array([A - z * mv, A + z * mv])
            
        elif flag == 'wald-cc': 
            Aci = np.array([A - (z * mv + 1 / (2 * N)), A + (z * mv + 1 / (2 * N))])
            
        elif flag == 'boot':
            # Simulate 'boot' with simple bootstrap percentile (because BOOTCI is not in Python)
                        
            # Aci = bootci(nboot,{@auc,data},'type','per')';
            # warning('BOOTCI function not available, resorting to simple percentile bootstrap in AUC.')
            
            A_boot = np.zeros(nboot)
            N_data = data.shape[0]
            for i in range(nboot):
               # MATLAB: ind = unidrnd(N,[N 1]);
               # np.random.randint(low, high, size)
               ind = np.random.randint(0, N_data, N_data) 
               # Here, we need to call auc which returns only A (nargout == 1)
               A_boot[i] = auc_value_only(data[ind, :])
               
            # MATLAB: prctile(A_boot,100*[alpha/2 1-alpha/2]);
            lower_perc = alpha / 2 * 100
            upper_perc = (1 - alpha / 2) * 100
            Aci = np.percentile(A_boot, [lower_perc, upper_perc])
            
        else:
            raise ValueError('Bad FLAG for AUC!')
            
    # If only one output is requested, we do not return Aci.
    # In Python, it is hard to detect nargout, so we return two outputs.
    return A, Aci

# --- Helper function for bootstrap recursive call (AUC value only) ---
def auc_value_only(data):
    if data.shape[1] != 2:
       raise ValueError('Incorrect input size in AUC!')

    m = np.sum(data[:, 0] > 0)
    n = np.sum(data[:, 0] <= 0)
    
    if m == 0 or n == 0:
        return 0.5
        
    R = tiedrank(data[:, 1])
    A = (np.sum(R[data[:, 0] > 0]) - (m**2 + m) / 2) / (m * n)
    A = np.clip(A, 0.0, 1.0)
    
    return A