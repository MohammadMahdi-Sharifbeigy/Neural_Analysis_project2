import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from .cbrewer import cbrewer

# load colormaps
# MATLAB: jet=colormap('jet');
jet_cmap_name = 'jet'
N_jet = 256 # Standard jet size in MATLAB
jet = cm.get_cmap(jet_cmap_name)(np.linspace(0, 1, N_jet))[:, :3] # Get N_jet colors (R, G, B)

# MATLAB: RdBu=cbrewer('div', 'RdBu', 11);
# MATLAB: RdYlGn=cbrewer('div', 'RdYlGn', 11);
try:
    RdBu = cbrewer('div', 'RdBu', 11)
    RdYlGn = cbrewer('div', 'RdYlGn', 11)
except NameError:
    # Fallback/Mock data if cbrewer is not defined or fails
    RdBu = np.array([[0.64, 0.00, 0.12], [0.84, 0.24, 0.17], [0.96, 0.38, 0.30], [0.96, 0.65, 0.51], [0.99, 0.86, 0.78], 
                     [0.97, 0.97, 0.97], [0.82, 0.90, 0.94], [0.60, 0.77, 0.87], [0.42, 0.61, 0.77], [0.22, 0.40, 0.67], 
                     [0.02, 0.19, 0.38]])
    RdYlGn = np.array([[0.65, 0.00, 0.12], [0.88, 0.22, 0.16], [0.99, 0.58, 0.33], [1.00, 0.81, 0.50], [0.94, 0.97, 0.73], 
                       [0.85, 0.95, 0.60], [0.66, 0.86, 0.52], [0.40, 0.74, 0.45], [0.15, 0.59, 0.35], [0.00, 0.41, 0.22], 
                       [0.00, 0.25, 0.00]])

# Define the new R, G, B references (p stands for prime)
# RdBu(1,:) is the darkest Red
Rp = RdBu[0, :]
# RdBu(end, :) is the darkest Blue
Bp = RdBu[-1, :]
# RdYlGn(end-2, :) is the third darkest green (Index 11-2 = 9, 0-based index 8)
# If RdYlGn is 11 colors, end-2 (index 9) is the third from the end (darkest green is index 10)
# This assumes the question meant the third darkest (or third lowest index from the end).
Gp = RdYlGn[-3, :] 
RGBp = np.vstack([Rp, Gp, Bp]).T # Must be 3x3 for projection (R' G' B' as columns)

# construct the new colormap based on the existing RGB values of jet
# Project the RGB values on your new basis
# MATLAB: newjet = jet*RGBp;
newjet = jet @ RGBp # Matrix multiplication in Python

# Clip values to [0, 1] as projection can result in values outside this range
newjet = np.clip(newjet, 0, 1)

# Store data in a structure/dictionary
cmap = {'jet': jet, 'newjet': newjet}
cnames = ['jet', 'newjet']

# --- Plot the RGB values ---
fh = plt.figure(figsize=(8, 6))
colors_rgb_label = ['r', 'g', 'b']
for iname_idx, cname in enumerate(cnames):
    # MATLAB: cnames{end-iname+1} -> plots in reverse order
    plot_name = cnames[len(cnames) - 1 - iname_idx]
    ax = fh.add_subplot(len(cnames), 1, iname_idx + 1)
    dat = cmap[plot_name]
    
    for icol in range(dat.shape[1]):
        ax.plot(dat[:, icol], color=colors_rgb_label[icol], linewidth=2)
    
    ax.set_title(f'"{plot_name}" in RGB plot')
    ax.tick_params(direction='out')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_xlabel('Index')
    ax.set_ylabel('RGB Value')

plt.tight_layout()

# --- Plot the colormaps ---
fh2 = plt.figure(figsize=(8, 4))
fh2.set_facecolor('white')

for iname_idx, cname in enumerate(cnames):
    F = cmap[cname]
    ncol = len(F)
    fg = 1.0 / ncol # geometrical factor
    
    # Define Y coordinates
    Y_base = (2 * iname_idx + 1) * 0.1
    Y_rect = np.array([0.1, 0, 0, 0.1]) # Height of the bar
    Y = Y_rect + Y_base
    
    for icol in range(ncol):
        # Define X coordinates
        X2 = np.array([0, 0, 1, 1]) * fg * icol + np.array([0, fg, fg, 0]) * fg
        
        # Fill the color segment
        plt.fill(X2, Y, color=F[icol, :], edgecolor='none')
        
    # Add text label
    plt.text(-0.05, Y_base + 0.05, cname, 
             transform=plt.gca().transAxes, 
             ha='right', va='center', 
             fontweight='bold', fontsize=10)
    
    # Set limits and hide axis
    plt.xlim([-0.4, 1])
    plt.axis('off')
    
# Final figure adjustments
plt.ylim([0.1, 1.05 * (2 * len(cnames) * 0.1 + 0.1)])
plt.show()