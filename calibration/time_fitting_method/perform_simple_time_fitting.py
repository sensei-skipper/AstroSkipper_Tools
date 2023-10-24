"""
Pickle data for several image datasets
"""
import numpy as np
import os
import matplotlib.pyplot as plt
from astropy.io import fits
import scipy
from scipy import interpolate
from scipy import optimize
import sys
from scipy.stats import sigmaclip
import copy
import pathlib
import pickle

from calibration_time_fitting import *

# Define 
nrows = 600
ncols = 3450
ss_os_slice = slice(3000, 3200)
ms_os_slice = slice(3200, 3450)
ss_slices = [slice(0, 1000), slice(1050, 3000)]
ms_slices = [slice(1000, 1050)]
gain=140

i_amp = 0
paths = [
         '../data/',
         '../data/',
        ]
file_strs = [
             'images_1m_wait_',                #lta2
             'images_20m_wait_',                #lta2
            ]
outdirs = ['../data/outdir_' + file_str[:-1] for file_str in file_strs]
nimageStarts = [
    1, 
    1, 
               ]
nimageEnds = [
    100,
    100,
]

# set up figs/axs for plots that use data from each image set
fig_ss_height_vs_os, axs_ss_height_vs_os = plt.subplots(1)
for outdir, path, file_str, nimageStart, nimageEnd in zip(outdirs, paths, file_strs, nimageStarts, nimageEnds):
    # 1. Get data
    print(outdir)
    data = get_data(outdir=outdir,   
              path=path,
              file_str=file_str,
              i_amp=i_amp,
              NROWS=nrows,
              NCOLS=ncols,
              nImageStart=nimageStart,
              nImageEnd=nimageEnd,
              overwrite=False)  
    # Get OS subtracted data and median single sample and multisample values per row
    os_sub_data = do_row_wise_os_subtraction(data, ss_slices, ms_slices, ss_os_slice, ms_os_slice)
    ss_os, ms_os = get_median_os_per_image(data, ss_os_slice, ms_os_slice, nSkipIm=0, nSkipRow=0)

#     ## 2. Plot per image behaviour
    plot_median_per_image(data, ss_slices, ms_slices, ss_os_slice, ms_os_slice, nSkipIm=0, nSkipRow=0, os_sub_data=os_sub_data, save=True, outdir=outdir, filestr=file_str)
    
#     ## 3. Fit line to height of transient vs ms_os
    ydata = np.median(np.median(os_sub_data[:, :, ss_slices[1].start:ss_slices[1].start+5], axis=-1), axis=-1)
    popt, _ = scipy.optimize.curve_fit(f=linear_func,
                                      xdata=ms_os,
                                      ydata=ydata)
    axs_ss_height_vs_os.scatter(ms_os, ydata,
                label=file_str + '| m={:0.2f}, b={:0.2f}'.format(popt[0], popt[1]),
               )
    axs_ss_height_vs_os.plot(ms_os, linear_func(ms_os, *popt))
    axs_ss_height_vs_os.set_ylabel('Height of SS transient')

      ## ------------------------------------------------------------------- ## 
      ## --- Plot line fit to scaling factor vs ms_os for each image set --- ##
#       ## ------------------------------------------------------------------- ##
    amplitudes = fit_amplitudes_per_image(os_sub_data, ms_slices[0], title='', NelectronVrange=1, outdir='', filestr='', save=False)
    popt, _ = scipy.optimize.curve_fit(f=linear_func,
                                       xdata=ms_os,
                                       ydata=amplitudes)
    axs_ss_height_vs_os.scatter(ms_os, amplitudes, label=file_str + '| m={:0.2f}, b={:0.2f}'.format(popt[0], popt[1]))
    axs_ss_height_vs_os.plot(ms_os, linear_func(ms_os, *popt))
#       ## ------------------------------------------------------------------- ##

    ## 4. Check that local transient does not change across rows
    subtract_local_median_from_image(os_sub_data=os_sub_data, im_num=10, ss_slices=ss_slices, ms_slices=ms_slices, save=True, outdir=outdir, filestr=file_str)
    
    ## 5. Look at subtraction of global median transient from median row of each image
    subtract_global_median_from_images(os_sub_data, ss_slices, ms_slices, save=True, outdir=outdir, filestr=file_str)
    
    ## 6. Look at subtraction of scaling factor * global median transient
    residuals_after_amp_fit(os_sub_data, ss_slices, ms_slices, ms_os, save=True, outdir=outdir, filestr=file_str)

    ## 7. Fit line to scaling factor vs ms os
    residuals_after_amp_fit(os_sub_data, ss_slices, ms_slices, ms_os, save=True, outdir=outdir, filestr=file_str)
    
#     ##
axs_ss_height_vs_os.legend()
axs_ss_height_vs_os.set_xlabel('MS OS')
axs_ss_height_vs_os.set_ylabel('Scaling Factor')
fig_ss_height_vs_os.savefig(outdir + '/' + 'height-of-ss-trans-vs-ms_os.png')
plt.show()

## Perform "regular" calibration and fitted calibration
bias_data = pickle.load(open('./outdir_images_1m_wait/images_1m_wait.pkl', "rb"))
image_data = pickle.load(open('./outdir_images_20m_wait/images_20m_wait.pkl', 'rb'))

# bias_data = pickle.load(open('./outdir_images_20m_wait/images_20m_wait.pkl', "rb"))
image = image_data[10]
biases = bias_data[:25]
perform_regular_calibration(bias_data=biases,
                            image=image,
                            gain=gain,
                            os_subtract_func=do_row_wise_os_subtraction,
                            ss_slices=ss_slices, 
                            ms_slices=ms_slices, 
                            ss_os_slice=ss_os_slice, 
                            ms_os_slice=ms_os_slice, 
                            save=True,
                            outdir='./',
                            filestr='')
# 
perform_fitted_calibration(bias_data=biases,
                            image=image,
                            gain=gain,
                            os_subtract_func=do_row_wise_os_subtraction,
                            ss_slices=ss_slices, 
                            ms_slices=ms_slices, 
                            ss_os_slice=ss_os_slice, 
                            ms_os_slice=ms_os_slice, 
                            save=True,
                            outdir='./',
                            filestr='')
