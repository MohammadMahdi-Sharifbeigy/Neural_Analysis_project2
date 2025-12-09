import matplotlib.pyplot as plt

def formatFigure(ax=None):
    """
    MATLAB: function formatFigure(h)
    
    Applies standard aesthetic settings to a Matplotlib Axes object 
    (or current axes if none is provided) to mimic MATLAB's style.
    
    Args:
        ax (matplotlib.axes.Axes, optional): The Axes object to format. 
                                            If None, uses plt.gca().
    """
    
    if ax is None:
        ax = plt.gca()

    # MATLAB: 'TickDir','out'
    ax.tick_params(direction='out')
    
    # MATLAB: 'Color', 'None' (Sets the background transparent/white)
    ax.patch.set_facecolor('None') 
    
    # MATLAB: 'box','off' (Removes top and right spines)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    
    # MATLAB: 'Fontname','Arial'
    # Applied to ticks and labels
    for label in (ax.get_xticklabels() + ax.get_yticklabels()):
        label.set_fontname('Arial')
        
    # MATLAB: 'TitleFontWeight', 'normal'
    ax.title.set_fontweight('normal')