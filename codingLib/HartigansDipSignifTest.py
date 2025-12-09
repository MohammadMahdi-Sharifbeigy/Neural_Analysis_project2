import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import uniform
from .HartigansDipTest import HartigansDipTest 

def is_number(s):
    """Helper to check if a string is a number (mimics isnumber in MATLAB)."""
    try:
        float(s)
        return True
    except ValueError:
        return False

def HartigansDipSignifTest(xpdf, nboot, **kwargs):
    """
    MATLAB: function [dip, p_value, xlow,xup]=HartigansDipSignifTest(xpdf,nboot,varargin)
    
    Calculates Hartigan's DIP statistic and its significance (p-value) 
    using a uniform distribution bootstrap sample.
    
    Args:
        xpdf (np.ndarray): Vector of sample values (empirical PDF).
        nboot (int): User-supplied sample size for bootstrap.
        **kwargs: Optional plotting arguments ('plot', plotid).
        
    Returns:
        tuple: (dip, p_value, xlow, xup) - Dip statistic, p-value, 
               lower and upper modal interval estimates.
    """
    
    # --- 0. Argument Parsing ---
    plotid = 6
    plothist = False
    
    # Parse varargin equivalent for plotting
    plot_args = kwargs.get('plot', False)
    if isinstance(plot_args, bool) and plot_args:
        plothist = True
        plotid = 1
    elif isinstance(plot_args, (int, float)):
        plothist = True
        plotid = int(plot_args)
    elif isinstance(plot_args, str) and is_number(plot_args):
        plothist = True
        plotid = int(float(plot_args))
        
    # --- 1. Calculate the DIP statistic from the empirical pdf ---
    try:
        # We only need the first 4 outputs: dip, xl, xu, ifault
        # MATLAB: [dip,xlow,xup, ifault, gcm, lcm, mn, mj]=HartigansDipTest(xpdf);
        dip, xlow, xup, ifault, _, _, _, _ = HartigansDipTest(xpdf)
    except NameError:
        raise NameError("The 'HartigansDipTest' function is required but not found in scope.")
        
    N = len(xpdf)
    
    # --- 2. Bootstrap Calculation ---
    
    # Calculate a bootstrap sample of size NBOOT of the dip statistic 
    # for a uniform pdf of sample size N
    boot_dip = np.zeros(nboot)
    for i in range(nboot):
        # MATLAB: unifpdfboot=sort(unifrnd(0,1,1,N));
        # uniform.rvs(loc=0, scale=1) generates uniform random numbers (0 to 1)
        unifpdfboot = np.sort(uniform.rvs(size=N)) 
        
        # MATLAB: [unif_dip]=HartigansDipTest(unifpdfboot); (Only need dip)
        unif_dip, _, _, _, _, _, _, _ = HartigansDipTest(unifpdfboot)
        boot_dip[i] = unif_dip
        
    # MATLAB: boot_dip=sort(boot_dip); (Already done implicitly by storing in pre-allocated array)
    boot_dip = np.sort(boot_dip)
    
    # MATLAB: p_value=sum(dip<boot_dip)/nboot;
    # Count how many bootstrap dips are greater than the observed dip
    p_value = np.sum(dip < boot_dip) / nboot
    
    # --- 3. Plot Boot-strap sample ---
    if plothist:
        # figure(plotid); clf;
        # [hy,hx]=hist(boot_dip);
        # bar(hx,hy,'k'); hold on;
        
        plt.figure(plotid)
        plt.clf()
        
        # Use Matplotlib's hist function to get histogram data
        hy, hx, _ = plt.hist(boot_dip, bins='auto', color='k', label='Bootstrap DIP Distribution')
        
        # plot([dip dip],[0 max(hy)*1.1],'r:');
        plt.plot([dip, dip], [0, np.max(hy) * 1.1], 'r:', linewidth=2, label=f'Observed DIP={dip:.4f}')
        
        plt.title('Bootstrap Significance Test for Hartigan\'s DIP')
        plt.xlabel('DIP Statistic')
        plt.ylabel('Frequency')
        plt.legend()
        plt.show()
        
    return dip, p_value, xlow, xup