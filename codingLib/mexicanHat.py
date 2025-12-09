import numpy as np

def mexicanHat(x, s, c):
    """
    MATLAB: function val = mexicanHat(x, s, c)
    
    Calculates a discrete approximation of the Mexican Hat wavelet 
    (negative second derivative of a Gaussian).
    
    Args:
        x (np.ndarray): Input vector (e.g., time or spatial points).
        s (float): Standard deviation (scale parameter) of the Gaussian.
        c (float): Center of the Gaussian.
        
    Returns:
        np.ndarray: The resulting wavelet filter/function.
    """
    
    # MATLAB: exponent = ((x-c).^2)./(2*s.^2);
    exponent = ((x - c)**2) / (2 * s**2)
    
    # Calculate the Gaussian kernel
    gaussian = np.exp(-exponent)
    
    # MATLAB: [0 -diff(diff((exp(-exponent)))) 0];
    # This computes the negative second discrete difference (derivative approximation)
    # and pads the result with zeros at the start and end.
    
    # First difference (dy/dx)
    diff1 = np.diff(gaussian)
    
    # Second difference (d^2y/dx^2)
    diff2 = np.diff(diff1)
    
    # Negative second difference: -diff2
    negative_diff2 = -diff2
    
    # Pad with zeros: [0, negative_diff2, 0]
    # np.pad is generally the cleanest way to pad
    val = np.pad(negative_diff2, (1, 1), 'constant', constant_values=0)
    
    return val