import numpy as np
import matplotlib.pyplot as plt

def jbfill(xpoints, upper, lower, color='b', edge='k', add=1, transparency=0.5):
    """
    MATLAB: function [fillhandle,msg]=jbfill(xpoints,upper,lower,color,edge,add,transparency)
    
    Fills the region between two curves (upper and lower) on a plot.
    
    Args:
        xpoints (np.ndarray): Horizontal data points (x-coordinates).
        upper (np.ndarray): The upper curve values.
        lower (np.ndarray): The lower curve values.
        color (str/tuple): Fill color. Default='b'.
        edge (str/tuple): Edge color. Default='k'.
        add (int): Flag to add to current plot (1) or make a new one (0). Default=1.
        transparency (float): Transparency (alpha) value (0 to 1). Default=0.5.
        
    Returns:
        tuple: (fillhandle, msg) - Fill handle (matplotlib Polygon) and error message.
    """
    
    xpoints = np.array(xpoints).flatten()
    upper = np.array(upper).flatten()
    lower = np.array(lower).flatten()
    
    if not (len(upper) == len(lower) and len(lower) == len(xpoints)):
        msg = 'Error: Must use the same number of points in each vector'
        return None, msg
        
    msg = ''
    
    # MATLAB: filled=[upper,fliplr(lower)];
    # Concatenate upper curve and reversed lower curve
    filled = np.concatenate((upper, np.flipud(lower)))
    
    # MATLAB: xpoints=[xpoints,fliplr(xpoints)];
    # Concatenate xpoints and reversed xpoints
    x_fill = np.concatenate((xpoints, np.flipud(xpoints)))
    
    # Handle 'add' flag
    if add:
        plt.gca() # Ensure current axes is established (implicit hold on)
    else:
        plt.figure() # Create a new figure
        
    # MATLAB: fillhandle=fill(xpoints,filled,color);
    # Matplotlib's fill returns a list of Polygon objects
    fill_handle = plt.fill(x_fill, filled, color=color, zorder=0)
    
    # MATLAB: set(fillhandle,'EdgeColor',edge,'FaceAlpha',transparency,'EdgeAlpha',transparency);
    # Apply settings to the first Polygon object in the list
    if fill_handle:
        poly = fill_handle[0]
        poly.set_edgecolor(edge)
        poly.set_facecolor(color) # Set face color explicitly again
        poly.set_alpha(transparency)
        # Note: MATLAB's set(..., 'EdgeAlpha', transparency) is not standard in fill().
        # We assume the user wants the standard behavior or use alpha for both.
        # Since 'EdgeColor' is set explicitly, we rely on the default edge alpha (1.0)
        # unless transparency is meant for both. Using poly.set_alpha(transparency) 
        # affects both face and edge.
        
    # The return value should be the handle list or the first handle
    return fill_handle, msg