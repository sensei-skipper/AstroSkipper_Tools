#!/usr/bin/env python
"""
Provides model fitting tools for SkipperImageROI
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.optimize import curve_fit
from scipy.interpolate import UnivariateSpline
from astropy.stats import sigma_clip
from scipy.signal import find_peaks
import warnings
from scipy.optimize import curve_fit
warnings.filterwarnings("ignore")
from scipy.interpolate import CubicSpline

def remove_outliers(x, y, threshold=3):
    """
    Remove outlier points from a dataset based on a z-score threshold.

    Parameters
    ----------
    x : array-like
        The x-values of the data.
    y : array-like
        The y-values of the data.
    threshold : float, optional
        The z-score threshold for outlier detection. Points with a z-score
        above this threshold are removed. Defaults to 3.

    Returns
    -------
    x_filtered : `numpy.ndarray`
        The x-values with outliers removed.
    y_filtered : `numpy.ndarray`
        The y-values with outliers removed.
    """
    # Calculate z-scores
    z_scores = np.abs((y - np.mean(y)) / np.std(y))

    # Filter out points with z-scores greater than the threshold
    filtered_indices = np.where(z_scores <= threshold)
    x_filtered = x[filtered_indices]
    y_filtered = y[filtered_indices]

    return x_filtered, y_filtered

def cubic_spline_fit(median_array,debugging=False):
    """
    Fit a cubic spline to a 1D array of median values.

    Parameters
    ----------
    median_array : array-like
        The 1D array of median (or similarly processed) values to be fitted.
    debugging : bool, optional
        If True, the function will plot the resulting cubic spline fit 
        overlaid on the filtered data points. Defaults to False.

    Returns
    -------
    numpy.ndarray
        An array of y-values corresponding to the cubic spline evaluated at 
        each x-position in the input data.

    Notes
    -----
    1. The function first applies a median filter (`signal.medfilt`) to
       `median_array`, resulting in `filtered_medians`.
    2. It then constructs a `UnivariateSpline` with a smoothing factor 
       set to `s=5.8e6` and spline order `k=3`.
    3. If `debugging` is True, it plots the original filtered data points
       and overlays the spline curve, allowing you to visually inspect
       the quality of the fit.
    4. Finally, the function returns the spline evaluated at integer 
       positions from 0 to len(`median_array`) - 1.
    """
    filtered_medians = signal.medfilt(median_array)
    spline = UnivariateSpline(np.arange(0,len(median_array),1), filtered_medians, k=3,s=5.8e6) #s=1.8e6)

    if debugging:
        fig, ax = plt.subplots(figsize=(10, 5)) 
        plt.rcParams['figure.dpi'] = 300
        plt.plot(np.arange(0,len(median_array),1), spline(np.arange(0,len(median_array),1)), color='red',linestyle='--',linewidth=3)
        
        plt.scatter(np.arange(0,len(median_array),1)[1:],filtered_medians[1:],s=100,marker='.',color='black')
        ax.tick_params(axis='both', which='major', labelsize=20)
        plt.xlabel("Columns",fontsize=25)
        plt.ylabel("Pixel Value (ADUs)",fontsize=25)
        plt.grid('both')
        plt.show()

    return spline(np.arange(0,len(median_array),1))


def natural_cubic_spline(y,debugging=False):
     """
    Fit a natural cubic spline to a 1D array of values and optionally remove outliers.

    Parameters
    ----------
    y : array-like
        The 1D array of values to be fitted with a natural cubic spline.
    debugging : bool, optional
        If True, the function will plot the computed spline alongside 
        the original data points for visual inspection. Defaults to False.

    Returns
    -------
    numpy.ndarray
        An array of y-values corresponding to the spline evaluated at each 
        integer position from 0 to len(y) - 1.

    Notes
    -----
    1. The function attempts to remove outlier points by first calling 
       `remove_outliers(x, y)`, where `x` is supposed to be an array of 
       integer positions from 0 to len(y) - 1. Make sure `x` is defined 
       before calling `remove_outliers` to avoid referencing it 
       prematurely.
    2. After outlier removal, the data is sorted by its x-values, and a 
       `CubicSpline` with `bc_type='natural'` is used to compute the 
       natural cubic spline fit.
    3. If `debugging=True`, the function plots:
       - The spline curve (red dashed line).
       - The original data points (black markers).
       This can help diagnose how well the spline matches the data.
    """

    x, y_filtered = remove_outliers(x, y)
    # Sort the data points by x values
    x = np.arange(0,len(y),1)
    sorted_indices = np.argsort(x)
    x_sorted = x[sorted_indices]
    y_sorted = y[sorted_indices]

    # Compute the cubic spline
    cs = CubicSpline(x_sorted, y_sorted, bc_type='natural')


    if debugging:
        
        fig, ax = plt.subplots(figsize=(10, 5)) 
        plt.plot(x, cs(x), color='red',linestyle='--',linewidth=3)
        plt.scatter(np.arange(0,len(y),1)[1:],y[1:],s=100,marker='.',color='black')
        ax.tick_params(axis='both', which='major', labelsize=20)
        plt.xlabel("Columns",fontsize=25)
        plt.ylabel("Pixel Value (ADUs)",fontsize=25)
        plt.grid('both')
        plt.show()

    return cs(x)

def find_peak_parameters(OS,left_bound,right_bound):
    """
    Identify the most prominent peak in a histogram of data.

    Parameters
    ----------
    OS : array-like
        The data for which the histogram is constructed and peaks are detected.
    left_bound : float
        The lower bound for the histogram bins.
    right_bound : float
        The upper bound for the histogram bins.

    Returns
    -------
    numpy.ndarray
        A one-dimensional NumPy array of length 1 containing the bin center 
        corresponding to the highest (most prominent) peak in the histogram. 
        If no peaks are found, returns an array containing a single value of 0.

    Notes
    -----
    1. The function constructs a histogram using Matplotlib's `plt.hist` with
       80 bins evenly spaced between `left_bound` and `right_bound`.
    2. It then uses `scipy.signal.find_peaks` to locate any peaks in the 
       histogram's bin counts. The peak with the maximum height is considered 
       the most prominent.
    3. The histogram figure is closed after plotting to prevent 
       displaying it during the process.
    """
    param = list()

    n, bins, patches = plt.hist(OS,bins=np.linspace(left_bound,right_bound,80), density=False, fc='orange',histtype='step')
    plt.close()
   
    peaks, _ = find_peaks(n, height=3, distance=4, width=0.6)

    if len(peaks)>0:
        most_prominent_peak_index = np.argmax(n[peaks])  
        param.append(bins[peaks[most_prominent_peak_index]])
    else:
        param.append(0)

    return np.array(param)

def gaussian(x,mu,sigma,A):
    """Gaussian"""
    return A*np.exp(-(x-mu)**2/2/sigma**2)

def SuperPosition(x,*params):
    """Superposition of Gaussians"""
    params = np.asarray(params)
    gaussians = list()
    ngauss = len(params)//3
    for i,p in enumerate(np.split(np.asarray(params),ngauss)):
        gauss = np.array(gaussian(x,*p))
        gaussians.append(gauss)
    return (np.array(gaussians).sum(axis=0))
       
def MultiGaussian(x,y,guess):
    """Fit a superposition of Gaussians"""
    params,cov=curve_fit(SuperPosition,x,y,p0=guess.flatten(),maxfev = 1000000)
    sigma=np.sqrt(np.diag(cov))
    return params, cov

def gaussian_fit(os_input_array,image,OSStart,OSSend,multi_sample_slices,serial=True):

     """
    Estimate and return an overscan baseline array via peak detection and optional masking.

    This function detects the most prominent peak of each row (in the serial case) or
    processes a 1D array (in the parallel case) and then applies a cubic spline to
    interpolate the detected or given values. The name "gaussian_fit" reflects the
    peak detection using Gaussian-like binning, though it ultimately uses a spline 
    for the final correction array.

    Parameters
    ----------
    os_input_array : array-like
        A 1D array of overscan values, used only if `serial=False`.
        When `serial=True`, this parameter is ignored.
    image : `numpy.ndarray`
        The 2D image data used in the serial case to detect peaks along each row.
    OSStart : int
        The starting column index for the overscan region in the serial case.
    OSSend : int
        The ending column index (exclusive) for the overscan region in the serial case.
    multi_sample_slices : list of slice or None
        In the parallel (non-serial) case, these slices are zeroed out (masked) 
        after the spline fit if not None. In the serial case, this parameter is 
        unused.
    serial : bool, optional
        If True, processes the image row by row in the specified overscan range
        (`OSStart:OSSend`) and detects peaks using `find_peak_parameters`.
        If False, the function treats `os_input_array` as the input data instead 
        of row-by-row image data. Defaults to True.

    Returns
    -------
    numpy.ndarray
        A 1D array of estimated overscan baseline values after peak detection, 
        outlier handling (via median if no peak is found), and final smoothing 
        using a cubic spline. The array length matches the number of rows 
        (for serial) or the length of `os_input_array` (for parallel).

    Notes
    -----
    1. **Serial Case**:  
       - Each row in `image` is sigma-clipped to remove outliers.  
       - The minimum and maximum pixel values in that row slice define histogram
         bounds for peak detection.  
       - If no valid peak is found, the median of that row's data is used instead.  
       - Finally, a 1D spline (`cubic_spline_fit`) is applied across all row-wise 
         peaks to generate a smooth overscan baseline array.

    2. **Parallel (Non-Serial) Case**:  
       - The function directly applies a spline to `os_input_array`.  
       - If `multi_sample_slices` is provided, it creates a mask array to zero out
         those regions after the spline is computed.
    """

    os_array = list()

    if serial:
        rows=image.shape[0]
        
        for r in range(rows):
            data = image[r:r+1,OSStart:OSSend]
            data = sigma_clip(data)
            
            left_bound = np.min(data.flatten())
            right_bound = np.max(data.flatten())
            
            mu=find_peak_parameters(data.flatten(),left_bound,right_bound)
            
            if mu[0]==0:
                mu = [np.median(data.flatten())]
            
            os_array.append(mu)

        os_array = cubic_spline_fit(np.concatenate(np.array(os_array)))
 
    else:

        os_array = cubic_spline_fit(os_input_array)

        if multi_sample_slices == None:
            os_array = os_array
        
        else:
            mask_array = np.ones(len(os_array),dtype=int)
            
            for zero_slice in multi_sample_slices:
                mask_array[zero_slice]=0
            
            os_array = mask_array*os_array
    
    return os_array
