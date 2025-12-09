import numpy as np
from scipy.ndimage import median_filter, convolve
from skimage.measure import label, regionprops
from skimage.draw import rectangle_perimeter
import warnings

def FastPeakFind(d, thres=None, filt=None, edg=3, res=1, fid=None):
    """
    MATLAB: function [cent, varargout]=FastPeakFind(d, thres, filt ,edg, res, fid)
    
    Finds peaks in noisy 2D images using local maxima (1 pixel resolution) 
    or weighted centroids (sub-pixel resolution).
    
    Args:
        d (np.ndarray): The 2D data raw image.
        thres (float, optional): Threshold to remove background. Default is a robust estimate.
        filt (np.ndarray, optional): Filter matrix used to smooth the image. Default is Gaussian.
        edg (int): Number > 1 for skipping edge pixels in local maxima method. Default=3.
        res (int): 1 for local maxima (default), 2 for weighted centroid (sub-pixel).
        fid (file object, optional): File handle to save peak positions.
        
    Returns:
        tuple: (cent, cent_map) or (cent,) 
               cent (np.ndarray): Peak coordinates (x1, y1, x2, y2, ...).
               cent_map (np.ndarray, optional): Binary matrix of peak positions (only for res=1).
    """
    
    # --- 0. Defaults and Input Preprocessing ---
    d = np.array(d)
    
    # I added this in case one uses imread (JPG\PNG\...).
    if d.ndim > 2 and d.shape[2] in [3, 4]:
        # Simple grayscale conversion (similar to MATLAB rgb2gray)
        d = np.dot(d[...,:3], [0.2989, 0.5870, 0.1140]).astype(d.dtype)

    if d.dtype in [np.float32, np.float64]:
        if np.max(d) <= 1:
            # Scale to fit uint16 range (0 to 65535)
            d = np.uint16(d * 2**16 / np.max(d) if np.max(d) > 0 else d)
        else:
            d = np.uint16(d) # Cast large float to uint16
    elif d.dtype == np.uint8:
        pass # Keep uint8 for efficient ops initially
    elif d.dtype != np.uint16:
        d = d.astype(np.uint16)

    # Threshold default (similar to max([min(max(d,[],1)) min(max(d,[],2))]))
    if thres is None:
        if d.size > 0:
            thres = max(np.min(np.max(d, axis=0)), np.min(np.max(d, axis=1)))
        else:
            thres = 0

    # Filter default (fspecial('gaussian', 7,1))
    if filt is None:
        # Create a basic Gaussian kernel (approximation of fspecial)
        sigma = 1.0
        size = 7
        x_g, y_g = np.mgrid[-size//2 + 1:size//2 + 1, -size//2 + 1:size//2 + 1]
        g = np.exp(-(x_g**2 + y_g**2) / (2 * sigma**2))
        filt = g / g.sum()

    # check thres is a scalar
    if np.isscalar(thres) is False:
        raise ValueError('Threshold has to be a scalar')
        
    savefileflag = fid is not None
    
    cent = np.array([])
    cent_map = np.zeros_like(d, dtype=int)
    
    if not np.any(d): # In case raw image is all zeros
        return cent, cent_map

    # --- 1. Pre-filtering and Thresholding ---
    
    # d = medfilt2(d,[3,3]); (Using SciPy's median_filter)
    d_filtered = median_filter(d, size=3)
    
    # Apply threshold (d=d.*uint(d>thres);)
    d_thres = d_filtered * (d_filtered > thres)
    
    if not np.any(d_thres):
        return cent, cent_map
    
    # smooth image (d=conv2(single(d),filt,'same'))
    # Use single precision for convolution if d is not already float
    d_smooth = convolve(d_thres.astype(np.float32), filt, mode='constant', cval=0.0)
    
    # Apply again threshold (d=d.*(d>0.9*thres);)
    # Must use float/single precision thres comparison
    d_processed = d_smooth * (d_smooth > 0.9 * thres)
    
    if not np.any(d_processed):
        return cent, cent_map

    # --- 2. Peak Finding Methods ---
    
    if res == 1: # Local maxima approach - 1 pixel resolution
        
        sd = d_processed.shape
        
        # d will be noisy on the edges, and also local maxima looks for nearest neighbors 
        # so edge must be at least 1. We'll skip 'edge' pixels.
        # MATLAB: [x, y]=find(d(edg:sd(1)-edg,edg:sd(2)-edg));
        
        # Get indices within the non-edge region where value > 0
        non_edge_slice = d_processed[edg-1:sd[0]-edg, edg-1:sd[1]-edg]
        x_local, y_local = np.where(non_edge_slice > 0)
        
        # Adjust indices back to original coordinates
        x_local += edg - 1
        y_local += edg - 1
        
        temp_cent = []
        for j in range(len(y_local)):
            x_j, y_j = x_local[j], y_local[j]
            
            val = d_processed[x_j, y_j]
            
            # Check if value is strictly greater than all 8 neighbors (MATLAB logic)
            is_local_max = True
            
            # Use slice to check neighbors (Faster than individual checks)
            neighbors = d_processed[x_j-1:x_j+2, y_j-1:y_j+2]
            # Exclude the center point itself (index 1, 1 in the 3x3 array)
            
            # Check if all neighbors are strictly less than the center (excluding center itself)
            # The check is `val > d_neighbor`. If any neighbor is >= val, it's not a strict local maximum.
            is_local_max = np.all(neighbors.flatten()[[0, 1, 2, 3, 5, 6, 7, 8]] < val)

            if is_local_max:
                # cent = [cent ;  y(j) ; x(j)]; (y, x order)
                temp_cent.extend([y_j + 1, x_j + 1]) # Convert to 1-based coordinates
                cent_map[x_j, y_j] += 1
                
        cent = np.array(temp_cent)
        
    elif res == 2: # Weighted centroid sub-pixel resolution
        
        # get peaks areas and centroids
        # MATLAB: stats = regionprops(logical(d),d,'Area','WeightedCentroid');
        
        # Create binary mask and label connected components
        d_binary = d_processed > 0
        labeled_img = label(d_binary)
        
        # Calculate region properties, including area and weighted centroid
        stats = regionprops(labeled_img, intensity_image=d_processed, extra_properties=[
            lambda region: region.weighted_centroid[::-1] # Convert (row, col) to (col, row)
        ])
        
        # Find reliable peaks (area <= mean + 2*std)
        areas = np.array([s.area for s in stats])
        if areas.size == 0:
            return cent, cent_map
            
        area_limit = np.mean(areas) + 2 * np.std(areas)
        rel_peaks_mask = areas <= area_limit
        
        # Extract centroids of reliable peaks (WeightedCentroid is (y, x) in MATLAB)
        temp_cent = []
        for s in np.array(stats)[rel_peaks_mask]:
            # WeightedCentroid is (row, col) in skimage, MATLAB expects (x, y) which is (col, row)
            # Add 1 for 1-based indexing
            temp_cent.extend([s.weighted_centroid[1] + 1, s.weighted_centroid[0] + 1])
            
        cent = np.array(temp_cent)
        cent_map = np.array([]) # Not supported for res=2
        
    else:
        raise ValueError("Parameter 'res' must be 1 or 2.")

    # --- 3. Output and File Saving ---
    
    if savefileflag:
        # MATLAB: fprintf(fid, '%f ', cent(:)); fprintf(fid, '\n');
        cent_reshaped = cent.reshape(2, -1).T
        np.savetxt(fid, cent_reshaped, fmt='%f', delimiter=' ', newline='\n')
    
    # Return binary mask if asked for (varargout in MATLAB)
    if res == 1:
        return cent, cent_map
    else:
        # res=2 does not return cent_map
        return cent

# Example usage (for testing - simplified):
# np.random.seed(42)
# test_image = np.zeros((50, 50))
# test_image[10:15, 10:15] = 100 # Peak 1
# test_image[30:35, 40:45] = 150 # Peak 2
# noise = 10 * np.random.randn(50, 50)
# test_image += noise
# cent_locmax, cent_map_locmax = FastPeakFind(test_image, thres=50, res=1)
# cent_centroid = FastPeakFind(test_image, thres=50, res=2)
# print("Local Maxima Centroids (y, x):", cent_locmax.reshape(-1, 2))
# print("Weighted Centroid Centroids (y, x):", cent_centroid.reshape(-1, 2))