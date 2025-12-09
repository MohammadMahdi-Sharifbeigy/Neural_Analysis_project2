import matplotlib.pyplot as plt
import matplotlib as mpl

def cleanFigure(ax=None):
    """
    MATLAB: set(gca,'TickDir','out', 'Color', 'None', 'box','off','Fontname','Arial', 'TitleFontWeight' , 'normal')
    
    Applies common cleanup settings to a Matplotlib Axes object 
    to mimic the aesthetic of the original MATLAB code.
    
    Args:
        ax (matplotlib.axes.Axes, optional): The Axes object to clean. 
                                            If None, uses plt.gca().
    """
    if ax is None:
        ax = plt.gca()

    # Set properties
    
    # TickDir: 'out'
    ax.tick_params(direction='out')
    
    # Color: 'None' (Sets the background transparent/white)
    ax.patch.set_facecolor('None') 
    
    # box: 'off' (Removes top and right spines)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    
    # Fontname: 'Arial'
    # Setting font properties globally or per element can be complex in Python.
    # We apply fontname to ticks and labels.
    for label in (ax.get_xticklabels() + ax.get_yticklabels()):
        label.set_fontname('Arial')
        
    # TitleFontWeight: 'normal'
    ax.title.set_fontweight('normal')

# Example usage (for testing):
# import numpy as np
# plt.plot(np.random.rand(10))
# cleanFigure()
# plt.title('Test Plot', fontweight='bold') # Check if normal is applied
# plt.show()