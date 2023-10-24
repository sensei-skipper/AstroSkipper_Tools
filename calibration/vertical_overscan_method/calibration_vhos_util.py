import numpy as np
import pickle
import matplotlib.pyplot as plt
from astropy.io import fits
import scipy
import os
import sys
# sys.path.append('astroskipper_calibration_tools/AstroSkipper_Tools/calibration/')
import copy
import pathlib

def plot_image(image, vmax, vmin=None, center=None, title=''):
    """
    Wrapper for plt.imshow with convenient defaults. Plots image from single amplifier.
    The default centers colorbar at median value of entire 2d image, with colorbar max/min at +/- vmax

    Args
    ----
    image (numpy array): 2d array from output of 
    vmax (float): Defines maximum of colorbar range relative to center.
    vmin (float): Defines minimum of colorbar range relative to center.
    center (float): Center of colorbar range.
    """
    if center is None:
        center = np.median(image)
    if vmin is None:
        vmin = vmax
    plt.imshow(image,
           aspect='auto',
            cmap='PRGn',
           vmin=center - vmin,
           vmax=center + vmax
          )
    plt.colorbar()
    plt.title(title)
    plt.show();
    
def get_image_params(fn):
    """
    Obtains relevant parameters from image header file located at 'fn'.
    Used in following functions: get_image

    Args
    ----
    fn (str): path to filename of image

    Returns
    -------
    nrows, ncols, nsamp, hos_start, vos_start
    """
    if '_ave.fits' in fn:
        header_index = 0
    elif '.fz' in fn:
        header_index = 1
    header = fits.open(fn)[header_index].header
    
    params = []
    for param in ['NROW', 'NCOL', 'NSAMP', 'CCDNCOL', 'CCDNROW']:
        val = int(header[param])
        if 'CCD' in param:
            val //= 2
        params.append(val)
    nrows, ncols, nsamp, hos_start, vos_start = params
    return nrows, ncols, nsamp, hos_start, vos_start, header

def get_image(fn):
    '''
    Args
    ----
    

    Returns
    -------
    image (): ndim = 3, shape=(i_amp, irow, icol)
    '''
    nrows, ncols, nsamp, hos_start, vos_start, header = get_image_params(fn)
    if '_ave.fits' in fn: # "_ave.fits" data starts at hdu index 0 
        i_amp_offset = 0
        ncols_true = ncols
    else: # '_all.fz' image data starts at index 1
        i_amp_offset = 1
        ncols_true = ncols
        ncols *= nsamp
    image = np.zeros(shape=(4, nrows, ncols))
    for i in range(4):
        image[i] = fits.open(fn)[i+i_amp_offset].data
    if ncols != ncols_true:
        image = image.reshape((4, nrows, ncols_true, nsamp))
        image = np.mean(image, axis=-1)
    return image, nrows, ncols_true, nsamp, hos_start, vos_start, header

def calibrate_image(fn, calibration_method='vh', ss_os_slice=None, ms_os_slice=None, ss_slices=None, ms_slices=None):
    '''
    Removes readout noise transients according to the calibration_method chosen. Assumes each ROI spans all rows.
    The default method (calibration_method='vh') removes row-wise and column-wise variation by first subtracting the mean of the horizontal overscan from the active area row by row then, second, subtracting the mean of the vertical overscan from the active area column by column.

    This function is made to be somewhat modular in that if a new calibration method is needed, one needs to define a function externally, associate it with a name (str) as an option for "calibration_method" argument, and add an if statement. 

    Args
    ----
    fn (str): filename of image
    calibration_method (str): Takes 'v' (vertical overscan subtraction; ie, removes column-wise variations), 'h' (horizontal), or 'vh' (horizontal then vertical)
    ss_os_slice (slice): A single slice of the form slice(ss_os_start, ss_os_stop) where ss_os_start is the column number at which the single sample overscan starts, and ss_os_stop is where it ends.
    ms_os_slice (slice): A single slice of the form slice(ms_os_start, ms_os_stop) where ms_os_start, ms_os_stop is the same as above.
    ss_slices (list of slices): Each slice in the list defines each single sample region. Each of these regions will undergo horizontal overscan corrections from the SS OS region.
    ms_slices (list of slices): Each slice in the list defines each multi sample region. Each of these regions will undergo horizontal overscan corrections using the MS OS region.
    '''
    image, nrows, ncols, nsamp, hos_start, vos_start, header = get_image(fn)
    args = [vos_start, hos_start, ss_os_slice, ms_os_slice, ss_slices, ms_slices]
    is_simple = all(x is None for x in args[2:]) # is_simple is True if the image has same number of samples for all pixels
    if calibration_method == 'v':
        args = [args[0]]
        if is_simple:
            calibration_func = doVOS_simple
        else:
            calibration_func = doVOS
    elif calibration_method == 'h':
        if is_simple:
            args = [args[1]]
            calibration_func = doHOS_simple
        else:
            args = args[1:]
            calibration_func = doHOS
    elif calibration_method == 'vh':
        if is_simple:
            args = args[:2]
            calibration_func = doVHOS_simple
        else:
            calibration_func = doVHOS
    calibrated_image = np.zeros_like(image)

    for i in range(4):
        calibrated_image[i] = calibration_func(image[i], *args)
    return calibrated_image


def get_data(outdir, path, file_str, i_amp, NROWS, NCOLS, nImageStart, nImageEnd, overwrite=False, verbose=True):
    '''Useful if you want to deal with many images (with same ROI pattern) and care about sorting images in chronological order. 
    Pickles and returns a 3 dimensional array with shape (nImages, nrows, ncols)
    
    Args
    -----
    outdir (str): path to directory where pickled data will be saved. Creates directory if does not exist
    path (str): path to files which contain images
    file_str (str): the filename which prefixes the image number  by lta 
        Ex. In `image_lta_35_ave.fits`, file_str == 'image_lta_' 
    i_amp (int): amplitude, 0-indexed if 
    NROWS (int): number of rows in image
    NCOLS (int): number of cols in iamge
    nImageStart (int): the file number given by lta of the first image
        ex. In file `image_lta_10_ave.fits`, 10 is the file number
    nImageEnd (int): the file number given by lta of the last image.
    overwrite (bool): whether or not to overwrite the existing pickle file if it exists
    '''
    pathlib.Path(outdir).mkdir(parents=True, exist_ok=True)
    out_fn = file_str[:-1] + '.pkl'
    outpath = outdir + '/' + out_fn
    try:
        data = pickle.load(open(outpath, "rb"))
#         print(f'{os.getcwd()}/{outpath} exists.')
        if overwrite:
            print('Overwriting...')
            raise
        print(f'Loading file from pickled data in {os.getcwd()}/{outpath}')
    except:
        if verbose:
            print(f'Writing {out_fn}...')
        track_read_in = []
        data = np.zeros(shape=((nImageEnd - nImageStart+1), NROWS, NCOLS))
        for filename in os.listdir(path):
            if verbose:
                print(f'Found {filename}')
            check_str = file_str
            if check_str in filename:
                num = filename[len(check_str):-len('_ave.fits')]
                try:
                    num = int(num) - nImageStart
                    track_read_in.append(num)
                    image = fits.open(name=path + filename)[i_amp].data
                    data[num] = np.copy(image)
                    if verbose:
                        print(f'{filename} is read in')
                except:
                    if verbose:
                        print(f'Did not read in {filename}')
                    continue
        pickle.dump(data, open(outpath, "wb"))
        print(f'Success! {out_fn} saved to {os.getcwd()}/{outpath}')
    return data
    
    
##################################################################################################################
################ Functions with suffix `_simple` assumes all pixels in images contain same number of samples ##################
################ Otherwise, function assumes image contains ROIs #######################
##################################################################################################################

def doHOS_simple(image, hos_start):
    '''Horizontal OS subtraction single image. Image must have same Nsamp for all pixels'''
    hos_image = np.copy(image)
    os_array = np.median(hos_image[:, hos_start+4:], axis=-1)
    hos_image -= os_array[:, np.newaxis]
    return hos_image

def doVOS_simple(image, vos_start):
    vos_image = np.copy(image)
    median_vos_row = np.median(vos_image[vos_start+4:], axis=0)
    vos_image -= median_vos_row
    return vos_image

def doVHOS_simple(image, vos_start, hos_start):
    hos_image = doHOS_simple(image, hos_start)
    vhos_image = doVOS_simple(hos_image, vos_start)
    return vhos_image

def doSVHOS_simple(image, vos_start, hos_start, shift):
    vhos_image = doVHOS_simple(image, hos_start, vos_start)
    svhos_image = doShift_simple(vhos_image, shift)
    return svhos_image

def doHOS(image, hos_start, ss_os_slice:slice, ms_os_slice:slice, ss_slices:list, ms_slices:list):
    hos_image = np.copy(image)
    ss_os_array = np.median(hos_image[:, ss_os_slice.start+4:ss_os_slice.stop], axis=-1) # needs to be offset so that taking median row doesn't give array of zeros
    ms_os_array = np.median(hos_image[:, ms_os_slice.stop-50:ms_os_slice.stop], axis=-1)
    for ss_slice in ss_slices:
        hos_image[:, ss_slice] -= ss_os_array[:, np.newaxis]
    for ms_slice in ms_slices:
        hos_image[:, ms_slice] -= ms_os_array[:, np.newaxis]
    hos_image[:, ss_os_slice] -= ss_os_array[:, np.newaxis]
    hos_image[:, ms_os_slice] -= ms_os_array[:, np.newaxis]
    return hos_image

def doVOS(image, vos_start):
    vos_image = doVOS_simple(image, vos_start)
    return vos_image

def doVHOS(image, vos_start, hos_start, ss_os_slice:slice, ms_os_slice:slice, ss_slices:list, ms_slices:list):
    hos_image = doHOS(image, hos_start, ss_os_slice, ms_os_slice, ss_slices, ms_slices)
    vhos_image = doVOS(hos_image, vos_start)
    return vhos_image

# tests
# np.all(do_hos(image, vos_start, hos_start) == doHOS_simple(image, hos_start))
# np.all(do_hos(light_image, vos_start, hos_start) == doHOS_simple(light_image, hos_start))
# np.all(do_vos(image, vos_start) == doVOS_simple(image, vos_start)), np.all(do_vos(light_image, vos_start) == doVOS_simple(light_image, vos_start))
# np.all(do_vos(hos_image, vos_start) == doVHOS_simple(image, hos_start, vos_start)), np.all(do_vos(hos_light_image, vos_start) == doVHOS_simple(light_image, hos_start, vos_start))
# np.all(do_vos(hos_image, vos_start)*2 == doSVHOS_simple(image, hos_start, vos_start, NSAMP=4))
# np.all(do_vos(hos_image, vos_start)*2 == doShift_simple(doVHOS_simple(image, hos_start, vos_start), NSAMP=4))
# np.all(do_vos(hos_light_image, vos_start) == doShift_simple(doVHOS_simple(light_image, hos_start, vos_start), NSAMP=1))
# np.all(doShift_simple(doVHOS_simple(light_image, hos_start, vos_start), NSAMP=1) == doSVHOS_simple(light_image, hos_start, vos_start, NSAMP=1))
# np.all(doShift_simple(doVHOS_simple(image, hos_start, vos_start), NSAMP=4) == doSVHOS_simple(image, hos_start, vos_start, NSAMP=4))
# np.all(doShift_simple(doVHOS_simple(image, hos_start, vos_start), NSAMP=4) == doSVHOS_simple(image, hos_start, vos_start, NSAMP=4))
