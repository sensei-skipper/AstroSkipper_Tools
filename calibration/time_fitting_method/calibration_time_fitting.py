import numpy as np
import os
import matplotlib.pyplot as plt
from astropy.io import fits
import scipy
from scipy import interpolate
from scipy import optimize
import sys
# sys.path.append('astroskipper_calibration_tools/AstroSkipper_Tools/calibration/')
# from calibration_tools import *
from scipy.stats import sigmaclip
import copy
import pathlib
import pickle

def save_figure(fig, outdir, filestr, plot_name):
    out_fn = outdir + '/' + filestr + plot_name + '.png'
    fig.savefig(out_fn)
    plt.close()

def linear_func(xs, m, b):
    ys = m*xs + b
    return ys

def plot_image(image, nElectronsVrange, title=''):
    plt.imshow(image,
           aspect='auto',
            cmap='PRGn',
           vmin=np.median(image)-nElectronsVrange,
           vmax=np.median(image)+nElectronsVrange
          )
    plt.colorbar()
    plt.title(title)
    plt.show();

def get_data(outdir, path, file_str, i_amp, NROWS, NCOLS, nImageStart, nImageEnd, overwrite=False, verbose=True):
    '''
    Pickles and returns data where data.shape=(nImages, nrows, ncols) and images are sorted by time.
    '''
    pathlib.Path(outdir).mkdir(parents=True, exist_ok=True)
    out_fn = file_str[:-1] + '.pkl'
    outpath = outdir + '/' + out_fn
    try:
        data = pickle.load(open(outpath, "rb"))
        print(f'{os.getcwd()}/{outpath} exists.')
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

def do_mean_os_subtraction(data, ss_slices, ms_slices, ss_os_slice, ms_os_slice):
    '''
    Takes data (output from 'get_data')
    '''
    def _do_mean_one_image(image):
        ss_os = np.mean(image[:, ss_os_slice])
        ms_os = np.mean(image[:, ms_os_slice])
        for ss_slice in ss_slices:
            image[:, ss_slice] -= ss_os
        for ms_slice in ms_slices:
            image[:, ms_slice] -= ms_os
        return image
    data = np.copy(data)
    if data.ndim==2:
        print('only one image in "data"')
        image = data
        os_sub_image = _do_mean_one_image(image)
        return os_sub_image
    else:
        os_sub_data = np.zeros_like(data)
        for i, image in enumerate(data):
            os_sub_data[i] = image
        return os_sub_data
        
def do_row_wise_os_subtraction(data, ss_slices:list, ms_slices:list, ss_os_slice:slice, ms_os_slice:slice):
    def _do_row_wise_one_image(im):
        os_sub_image = np.zeros(shape=(data.shape[-2], data.shape[-1]))
        for jj, row in enumerate(im):
                ss_os_row = np.median(row[ss_os_slice.start+4:ss_os_slice.stop])
                ms_os_row = np.median(row[ms_os_slice.start+80:ms_os_slice.stop])
                for ss_slice in ss_slices:
                    row[ss_slice] -= ss_os_row
                for ms_slice in ms_slices:
                    row[ms_slice] -= ms_os_row
                row[ss_os_slice] -= ss_os_row
                row[ms_os_slice] -= ms_os_row
                os_sub_image[jj] = row
        return os_sub_image
    data = np.copy(data)
    if data.ndim==2: # if input is just one image
        image = data
        os_sub_image = _do_row_wise_one_image(image)
        return os_sub_image
    else: # else, iterate over each image
        os_sub_data = np.zeros_like(data)
        for i, image in enumerate(data):
            os_sub_image = np.zeros(shape=(data.shape[-2], data.shape[-1]))
            os_sub_data[i] = _do_row_wise_one_image(image)
    return os_sub_data

def get_median_os_per_image(data, ss_os_slice, ms_os_slice, nSkipIm=0, nSkipRow=0):
    ss_os = np.median(np.median(data[nSkipIm:, nSkipRow:, ss_os_slice], axis=-1), axis=-1)
    ms_os = np.median(np.median(data[nSkipIm:, nSkipRow:, ms_os_slice], axis=-1), axis=-1)
    return ss_os, ms_os

def plot_median_per_image(data, ss_slices, ms_slices, ss_os_slice, ms_os_slice, nSkipIm=0, nSkipRow=0, os_sub_data=None, save=False, outdir=None, filestr=None):
    ss_os, ms_os = get_median_os_per_image(data, ss_os_slice, ms_os_slice, nSkipIm, nSkipRow)
    ss_region = ss_slices[1]
    ms_region = ms_slices[0]
    
    raw_ss_close = np.median(np.median(data[nSkipIm:, nSkipRow:, ss_region.start:ss_region.start+5], axis=-1), axis=-1)
    raw_ms_close = np.median(np.median(data[nSkipIm:, nSkipRow:, ms_region.start:ms_region.start+5], axis=-1), axis=-1)
    if os_sub_data is None:
        os_sub_data = do_row_wise_os_subtraction(data, ss_slices, ms_slices, ss_os_slice, ms_os_slice)

    fig, axs = plt.subplots(2, 3, figsize=(25,10), sharex=True)
    
    axs[0,0].set_title('OS Single Sample')
    axs[0,0].plot(np.arange(len(ss_os)), ss_os, color='grey', label='OS')
    axs[1,0].set_title('OS Multi Sample')
    axs[1,0].plot(np.arange(len(ms_os)), ms_os, color='grey', label='OS')

    axs[0,1].set_title('Single Sample pixels (close)')
    axs[0,1].plot(np.arange(len(raw_ss_close)), raw_ss_close, color='orange', label='OS')
    axs[1,1].set_title('Multi Sample pixels (close)')
    axs[1,1].plot(np.arange(len(raw_ms_close)), raw_ms_close, color='lightgreen')

    axs[0,2].set_title('OS subtracted single sample (close)')
    axs[0,2].plot(np.arange(len(os_sub_data)), np.median(np.median(os_sub_data[:, :, ss_region.start:ss_region.start+5], axis=-1), axis=-1), color='orange')
    axs[1,2].set_title('OS subtracted multi sample (close)')
    axs[1,2].plot(np.arange(len(os_sub_data)), np.median(np.median(os_sub_data[:, :, ms_region.start:ms_region.start+5], axis=-1), axis=-1), color='lightgreen')

    for i in range(3):
        axs[1,i].set_xlabel('Image Number')
    fig.tight_layout()
    if save:
        save_figure(fig, outdir, filestr, 'image-by-image-plot')
        
    fig, axs = plt.subplots(2, 3, figsize=(25,10), sharex=True)
    
    axs[0,0].set_title('OS Single Sample')
    axs[0,0].plot(np.arange(len(ss_os)), ss_os, color='grey', label='OS')
    axs[1,0].set_title('OS Multi Sample')
    axs[1,0].plot(np.arange(len(ms_os)), ms_os, color='grey', label='OS')

    axs[0,1].set_title('Single Sample pixels (close)')
    axs[0,1].plot(np.arange(len(raw_ss_close)), raw_ss_close, color='orange', label='OS')
    axs[1,1].set_title('Multi Sample pixels (close)')
    axs[1,1].plot(np.arange(len(raw_ms_close)), raw_ms_close, color='lightgreen')

    axs[0,2].set_title('OS subtracted single sample (close)')
    axs[0,2].plot(np.arange(len(os_sub_data)), np.median(np.median(os_sub_data[:, :, ss_region.start:ss_region.start+5], axis=-1), axis=-1), color='orange')
    axs[1,2].set_title('OS subtracted multi sample (close)')
    axs[1,2].plot(np.arange(len(os_sub_data)), np.median(np.median(os_sub_data[:, :, ms_region.start:ms_region.start+5], axis=-1), axis=-1), color='lightgreen')

    for i in range(3):
        axs[1,i].set_xlabel('Image Number')
    fig.tight_layout()
    if save:
        save_figure(fig, outdir, filestr, 'image-by-image-plot')
        
def subtract_local_median_from_image(os_sub_data, im_num, ss_slices, ms_slices, save=True, outdir=None, filestr=None):
    nplots = len(ms_slices)+len(ss_slices)
    fig, axs = plt.subplots(1, nplots, figsize=(6*nplots, 5), sharey=True)

    median_row = np.median(os_sub_data[im_num, :, :], axis=0)
    image_minus_trans = np.copy(os_sub_data[im_num])
    
    i=0
    for ss_slice in ss_slices:
        image_minus_trans[:, ss_slice] -= median_row[ss_slice]
        im = axs[i].imshow(image_minus_trans[:, ss_slice],
                           aspect='auto',
                           vmin=-140*2,
                           vmax=140*2,
                           extent=[ss_slice.start, ss_slice.stop, len(image_minus_trans), 0]
                           )
        axs[i].set_title('SS')
        fig.colorbar(mappable=im, ax=axs[i])
        i+=1
    for ms_slice in ms_slices:
        image_minus_trans[:, ms_slice] -= median_row[ms_slice]
        im = axs[i].imshow(image_minus_trans[:, ms_slice],
                           aspect='auto',
                           vmin=-140,
                           vmax=140,
                           extent=[ms_slice.start, ms_slice.stop, len(image_minus_trans), 0]
                           )
        axs[i].set_title('MS')
        fig.colorbar(mappable=im, ax=axs[i])
        i+=1     
    for i in range(nplots):
        axs[i].set_xlabel('Col number')
        axs[i].set_ylabel('Row number')    
    fig.tight_layout()
    if save:
        save_figure(fig, outdir, filestr, 'residuals-after-subtracting-local-transient')

def subtract_global_median_from_images(os_sub_data, ss_slices, ms_slices, save=True, outdir=None, filestr=None):
    median_row_per_image = np.copy(np.median(os_sub_data, axis=1))
    global_median_row = np.copy(np.median(median_row_per_image, axis=0))
    
    def plot_region(med_row_per_im, glob_med_row, region_slice, title, NelectronVrange):
        fig, ax = plt.subplots(1)
        # get median transient per row
        im = ax.imshow(med_row_per_im[:, region_slice],
                   aspect='auto',
                   extent=[region_slice.start, min(region_slice.stop, region_slice.start+500), len(os_sub_data), 0]
                  )
        ax.set_xlabel('Col number')
        ax.set_ylabel('Image number (median row)')
        ax.set_title(title)
        fig.colorbar(mappable=im, ax=ax)
        fig.tight_layout()
        if save:
            save_figure(fig, outdir, filestr, title + '-stacked-median-row-of-each-image-vs-column')

        res = med_row_per_im[:, region_slice] - glob_med_row[region_slice]
        fig, ax = plt.subplots(1)
        im = ax.imshow(res,
                      aspect='auto',
                      extent=[region_slice.start, min(region_slice.stop, region_slice.start+500), len(os_sub_data), 0],
                      cmap='PRGn',
                      vmin=-140*NelectronVrange,
                      vmax=140*NelectronVrange
              )
        ax.set_title(title + ' Residuals')
        fig.colorbar(mappable=im, ax=ax)
        fig.tight_layout()
        if save:
            save_figure(fig, outdir, filestr, title + '-residuals-after-subtracting-global-transient')

    for ss_slice in ss_slices[1:]: # skip first b/c has vertical clock transient
        plot_region(med_row_per_im=median_row_per_image, glob_med_row=global_median_row, region_slice=ss_slice, title='SS', NelectronVrange=1)
        
    for ms_slice in ms_slices: 
        plot_region(med_row_per_im=median_row_per_image, glob_med_row=global_median_row, region_slice=ms_slice, title='MS', NelectronVrange=.5)
    return
     
def fitting_amp(ave_transient, A):
    return A * ave_transient

def residuals_after_amp_fit(os_sub_data, ss_slices, ms_slices, ms_os, save=True, outdir=None, filestr=None):
    median_row_per_image = np.copy(np.median(os_sub_data, axis=1))
    global_median_row = np.copy(np.median(median_row_per_image, axis=0))
    
    def get_amps(med_row_per_im, glob_med_row, region_slice, title, NelectronVrange):
        amplitudes = np.empty(shape=(med_row_per_im[:, region_slice].shape[0]))
        pcovs = np.empty(shape=(med_row_per_im[:, region_slice].shape[0]))
        # xs = np.arange(transient_per_image.shape[1])
        residuals = np.empty(shape=(med_row_per_im[:, region_slice].shape[0], med_row_per_im[:, region_slice].shape[1]))

        for i, transient in enumerate(med_row_per_im[:, region_slice]):
            popt, pcov = scipy.optimize.curve_fit(fitting_amp,
                                                  xdata=glob_med_row[region_slice.start:region_slice.start+300],
                                                  ydata=med_row_per_im[i, region_slice.start:region_slice.start+300],
                                                  p0=(1),
                                                  )
            amplitudes[i] = popt
            pcovs[i] = pcov
            residuals[i] = med_row_per_im[i, region_slice] - popt*glob_med_row[region_slice]
        fig, ax = plt.subplots(1)
        im = ax.imshow(residuals, aspect='auto',
               cmap='PRGn',
               vmin=np.mean(residuals)-140*NelectronVrange,
               vmax=np.mean(residuals)+140*NelectronVrange
              )
        ax.set_ylabel('Image number (median row)')
        ax.set_title(title + ' Residuals')
        fig.colorbar(mappable=im, ax=ax)
        fig.tight_layout()
        if save:
            save_figure(fig, outdir, filestr, title + '-residuals-after-subtracting-scaling-fit')
        plt.close()
        
        
        return amplitudes
    
    for i, ss_slice in enumerate(ss_slices):
        amps = get_amps(med_row_per_im=median_row_per_image, glob_med_row=global_median_row, region_slice=ss_slice, title='SS', NelectronVrange=1)
        interpolator = fit_scaling_factor_vs_ms_os(amps=amps, ms_os=ms_os, title=f'SS-Region{i+1}', save=save, outdir=outdir, filestr=filestr)
    for i, ms_slice in enumerate(ms_slices): 
        amps = get_amps(med_row_per_im=median_row_per_image, glob_med_row=global_median_row, region_slice=ms_slice, title='MS', NelectronVrange=.5)
        interpolator = fit_scaling_factor_vs_ms_os(amps=amps, ms_os=ms_os, title=f'MS-Region{i+1}', save=save, outdir=outdir, filestr=filestr)
    return

def fit_amplitudes_per_image(os_sub_data, region_slice, title, NelectronVrange, outdir, filestr, save=True):
    med_row_per_im = np.median(os_sub_data, axis=1)
    glob_med_row = np.median(med_row_per_im, axis=0)
    amplitudes = np.empty(shape=(med_row_per_im[:, region_slice].shape[0]))
    residuals = np.empty(shape=(med_row_per_im[:, region_slice].shape[0], med_row_per_im[:, region_slice].shape[1]))

    for i, transient in enumerate(med_row_per_im[:, region_slice]): # for each image
        popt, pcov = scipy.optimize.curve_fit(fitting_amp,
                                              xdata=glob_med_row[region_slice],
                                              ydata=med_row_per_im[i, region_slice],
                                              p0=(1),
                                              )
        amplitudes[i] = popt
#         pcovs[i] = pcov
        residuals[i] = med_row_per_im[i, region_slice] - popt*glob_med_row[region_slice]
    fig, ax = plt.subplots(1)
    im = ax.imshow(residuals, aspect='auto',
           cmap='PRGn',
           vmin=np.mean(residuals)-NelectronVrange,
           vmax=np.mean(residuals)+NelectronVrange
          )
    ax.set_ylabel('Image number (median row)')
    ax.set_title(title + ' Residuals')
    fig.colorbar(mappable=im, ax=ax)
    fig.tight_layout()
    if save:
        save_figure(fig, outdir, filestr, title + '-residuals-after-subtracting-scaling-fit')
    plt.close()
    
    return amplitudes

def fit_scaling_factor_vs_ms_os(amps, ms_os, title, save, outdir, filestr):
    popt_amp, pcov = scipy.optimize.curve_fit(linear_func,
                                          xdata=ms_os,
                                          ydata=amps,
                                          p0=(1, 1),
                                          )

    xs = ms_os
    ys = linear_func(ms_os, *popt_amp)
    sorted_indices = np.argsort(xs)
    xs = xs[sorted_indices]
    ys = ys[sorted_indices]

    j=-1
    for i in range(len(xs)-1):
        j+=1
        if xs[j] == xs[j+1]:
            xs = np.delete(xs, j)
            ys = np.delete(ys, j)
            j-=1

    assert all(i < j for i, j in zip(xs, xs[1:]))

    interpolator = scipy.interpolate.InterpolatedUnivariateSpline(x=xs, y=ys)

    fig, ax = plt.subplots(1)
    ax.scatter(ms_os, amps, c=np.arange(len(amps)))
    ax.scatter(ms_os, linear_func(ms_os, *popt_amp), color='black', edgecolor='red', label='fit')
    ax.plot(ms_os, interpolator(ms_os), label='interp', color='red')
    ax.set_ylabel('Scaling factor')
    ax.set_xlabel('MS OS')
    ax.legend()
    save_figure(fig, outdir, filestr, title + 'amplitude-fitting')
    return interpolator
   
    
    
####
# (1) get linear fit and interpolating function between scaling factor fit vs ms_os for bias images for each transient, then use interpolating function --- DONE
# (2) subtract each transient (1-3) from each bias image and take median for master bias --- DONE
    # for each image in bias_data
        # for each slice in ss_slices and ms_slices:
            # scaling_factor = interpolator_slice(ms_os)
            # image[:, slice] -= scaling_factor * median_transient[slice]
    # (3) subtract transient (1-3) from observation image
# (4) subtract master bias from observation image
# (5) Plot residuals and median row of residuals

def perform_fitted_calibration(bias_data, image, gain, os_subtract_func, ss_slices, ms_slices, ss_os_slice, ms_os_slice, save=True, outdir='./', filestr=''):
    # overscan subtract each image and convert signal to electrons
    os_sub_bias_data = os_subtract_func(bias_data, ss_slices, ms_slices, ss_os_slice, ms_os_slice)
    os_sub_image = os_subtract_func(image, ss_slices, ms_slices, ss_os_slice, ms_os_slice)
    # convert ADU to electrons
    os_sub_bias_data = os_sub_bias_data/gain
    os_sub_image = os_sub_image/gain
    # get ms_os per image for fitting
    _, ms_os_arr = get_median_os_per_image(os_sub_bias_data, ss_os_slice, ms_os_slice)
    # first take global median row of all biases
    bias_median_row = np.median(np.median(os_sub_bias_data[:, :, :], axis=1), axis=0)
    # get interpolators
    interpolators = []
    amplitudes = []
    for i, ss_slice in enumerate(ss_slices):
        amps = fit_amplitudes_per_image(os_sub_data=os_sub_bias_data,
                                        region_slice=slice(ss_slice.start, ss_slice.start+300),
                                        title=f'SS-Region-{i+1}',
                                        NelectronVrange=1,
                                        outdir=outdir,
                                        filestr=filestr,
                                        save=save)
        amplitudes.append(np.copy(amps))
        interpolator = fit_scaling_factor_vs_ms_os(amps=amps, ms_os=ms_os_arr, title=f'SS-Region-{i+1}', save=save, outdir=outdir, filestr=filestr)
        interpolators.append(interpolator)
    for i, ms_slice in enumerate(ms_slices):
        amps = fit_amplitudes_per_image(os_sub_data=os_sub_bias_data,
                                        region_slice=ms_slice,
                                        title=f'MS-Region-{i+1}',
                                        NelectronVrange=.5,
                                        outdir=outdir,
                                        filestr=filestr,
                                        save=save)
        amplitudes.append(np.copy(amps))
        interpolator = fit_scaling_factor_vs_ms_os(amps=amps, ms_os=ms_os_arr, title=f'MS-Region-{i+1}', save=save, outdir=outdir, filestr=filestr)
        interpolators.append(interpolator)
    # subtract each transient from all rows of each bias and construct master bias
    for i in range(len(os_sub_bias_data)):
        j = 0
        ms_os = ms_os_arr[i]
        for region_slice in np.concatenate((np.array(ss_slices), np.array(ms_slices))):
            scaling_factor = interpolators[j](ms_os)
#             plt.plot(os_sub_bias_data[i, 500, region_slice])
#             plt.plot(scaling_factor*bias_median_row[region_slice])
#             plt.show();
            os_sub_bias_data[i, :, region_slice] -= scaling_factor * bias_median_row[region_slice]
            j += 1
    master_bias = np.median(os_sub_bias_data, axis=0)
    j = 0
    # subtract each transient from all rows of observation image
    image_ms_os = np.median(os_sub_image[:, ms_os_slice])
    for region_slice in np.concatenate((np.array(ss_slices), np.array(ms_slices))):
        scaling_factor = interpolators[j](image_ms_os)
#         plt.plot(os_sub_image[1, region_slice])
#         plt.plot(scaling_factor * bias_median_row[region_slice])
#         plt.show();
        os_sub_image[:, region_slice] -= scaling_factor * bias_median_row[region_slice]
        j += 1
    image = os_sub_image
    
    # do master bias subtraction
    res = image - master_bias
        
    # Plots!
    
    # plot amp vs ms_os
#     fig, ax = plt.subplots()
#     ax.scatter(ms_os_arr,
#                 amplitudes[1],
#                 c=np.arange(len(amplitudes[1])), label='measured')
#     plt.scatter(mast_ms_os_list, mast_amp_list, label='estimated (bias)', marker='^', color='red')
#     plt.scatter(test_ms_os, test_amp, label='estimated (image)', marker='X', color='purple', edgecolor='black', s=300)
#     plt.scatter(ms_os[test_index], ss_amps[test_index], label='measured (image)', marker='X', color='yellow', edgecolor='black', s=200)
#     plt.legend()
#     plt.show();
    
    fig, ax = plt.subplots(1)
    im = ax.imshow(master_bias,
              aspect='auto',
                   cmap='PRGn',
               vmin=-3.5,
               vmax=3.5,
              )
    ax.set_title("Fitted Calibration - Master bias")
    fig.colorbar(mappable=im, ax=ax)
    if save:
        save_figure(fig, outdir, filestr, 'fitted-calibration-master-bias')
    
    fig, ax = plt.subplots(1)
    im = ax.imshow(image,
              aspect='auto',
                   cmap='PRGn',
               vmin=-3.5,
               vmax=3.5,
              )
    ax.set_title("'Fitted' Calibration - Observation Image")
    fig.colorbar(mappable=im, ax=ax)
    if save:
        save_figure(fig, outdir, filestr, 'fitted-calibration-observation-image')
    
    fig, ax = plt.subplots(1)
    im = ax.imshow(res,
              aspect='auto',
                   cmap='PRGn',
               vmin=-1,
               vmax=1,
              )
    ax.set_title("'Fitted' Calibration - Residuals")
    fig.colorbar(mappable=im, ax=ax)
    if save:
        save_figure(fig, outdir, filestr, 'fitted-calibration-residual-image')
    
    ss_vals_1 = res[:, :1000].flatten()
    ss_vals_2 = res[:, 1050:3000].flatten()
    ss_vals = np.concatenate((ss_vals_1, ss_vals_2))
    ms_vals = res[:, 1000:1050].flatten()
    print(f'sigma_ss = {np.std(ss_vals)}\n sigma_ms = {np.std(ms_vals)}')
    print(f'sigma_ss_1 = {np.std(ss_vals_1)}\n sigma_ss_2 = {np.std(ss_vals_2)}')
    
    fig, axs = plt.subplots(1, 2, figsize=(13,5))
    axs[0].plot(np.median(res, axis=0)[:1000])
    axs[0].plot(np.arange(1050,3000), np.median(res, axis=0)[1050:3000], color='C0')    
    axs[0].set_title('SS')
    axs[1].set_title('MS')
    axs[1].plot(np.arange(1000, 1050), np.median(res, axis=0)[1000:1050], color='C2')    
    for i in range(2):
        axs[i].set_xlabel('Column number')
        axs[i].set_ylabel('e- count')
    fig.suptitle("'Fitted' Calibration - Median row of residuals")
    if save:
        save_figure(fig, outdir, filestr, 'fitted-calibration-median-row-of-residuals')
    return res    
    
def perform_regular_calibration(bias_data, image, gain, os_subtract_func, ss_slices, ms_slices, ss_os_slice, ms_os_slice, save=True, outdir=None, filestr=None):
    os_sub_bias_data = os_subtract_func(bias_data, ss_slices, ms_slices, ss_os_slice, ms_os_slice)
    os_sub_image = os_subtract_func(image, ss_slices, ms_slices, ss_os_slice, ms_os_slice)

    os_sub_bias_data = os_sub_bias_data/gain
    os_sub_image = os_sub_image/gain
    
    master_bias = np.median(os_sub_bias_data, axis=0)
    res = os_sub_image - master_bias

    fig, ax = plt.subplots(1)
    im = ax.imshow(master_bias,
              aspect='auto',
                   cmap='PRGn',
               vmin=-3.5,
               vmax=3.5,
              )
    ax.set_title("'Regular' Calibration - Master bias")
    fig.colorbar(mappable=im, ax=ax)
    if save:
        save_figure(fig, outdir, filestr, 'regular-calibration-master-bias')
    
    fig, ax = plt.subplots(1)
    im = ax.imshow(os_sub_image,
              aspect='auto',
                   cmap='PRGn',
               vmin=-3.5,
               vmax=3.5,
              )
    ax.set_title("'Regular' Calibration - Observation Image")
    fig.colorbar(mappable=im, ax=ax)
    if save:
        save_figure(fig, outdir, filestr, 'regular-calibration-observation-image')
    
    fig, ax = plt.subplots(1)
    im = ax.imshow(res,
              aspect='auto',
                   cmap='PRGn',
               vmin=-1,
               vmax=1,
              )
    ax.set_title("'Regular' Calibration - Residuals")
    fig.colorbar(mappable=im, ax=ax)
    if save:
        save_figure(fig, outdir, filestr, 'regular-calibration-residual-image')
    
    ss_vals_1 = res[:, :1000].flatten()
    ss_vals_2 = res[:, 1050:3000].flatten()
    ss_vals = np.concatenate((ss_vals_1, ss_vals_2))
    ms_vals = res[:, 1000:1050].flatten()
    print(f'sigma_ss = {np.std(ss_vals)}\n sigma_ms = {np.std(ms_vals)}')
    print(f'sigma_ss_1 = {np.std(ss_vals_1)}\n sigma_ss_2 = {np.std(ss_vals_2)}')
    
    fig, axs = plt.subplots(1, 2, figsize=(13,5))
    axs[0].plot(np.median(res, axis=0)[:1000])
    axs[0].plot(np.arange(1050,3000), np.median(res, axis=0)[1050:3000], color='C0')    
    axs[0].set_title('SS')
    axs[1].set_title('MS')
    axs[1].plot(np.arange(1000, 1050), np.median(res, axis=0)[1000:1050], color='C2')    
    for i in range(2):
        axs[i].set_xlabel('Column number')
        axs[i].set_ylabel('e- count')
    fig.suptitle("'Regular' Calibration - Median row of residuals")
    if save:
        save_figure(fig, outdir, filestr, 'regular-calibration-median-row-of-residuals')
    return res
    
