import numpy as np
import os
import matplotlib.pyplot as plt
from astropy.io import fits
from scipy import interpolate
from scipy import optimize
import sys
sys.path.append('/home/rhur/Projects/skipperRegionalSelection/astroskipper_calibration_tools/AstroSkipper_Tools/calibration/')
from calibration_tools import *

path_to_cal15 = '/home/rhur/Projects/skipperRegionalSelection/data/astroskipper_images/new/N_15/calibration_image_1_N_15/'
path_to_cal500 = '/home/rhur/Projects/skipperRegionalSelection/data/astroskipper_images/new/N_500/calibration_image_1_N_500/'
plot_outdir = './astroskipper_dataset1_plots/'
centers = np.array([401, 402, 404, 407, 411, 416, 422, 429, 437, 446, 456, 467, 479, 492, 506, 521, 537, 554, 572, 591, 611, 636, 666, 701, 741, 786, 836, 891, 951, 1016, 1086, 1161, 1241, 1326, 1416, 1511, 1611, 1721, 1841, 1971, 2111, 2261, 2461, 2711, 3011, 3361, 3761, 4211])
centers -= 1
centers -= 200
centers = centers[1:]
file_strs_list = ['image', 'ave']
i_amp=0

calibration15 = BaselineCalibration(path_to_cal_image_dir=path_to_cal15,
                                           Nsamp=15,
                                           centers=centers,
                                           cal1_file_strs_list=file_strs_list,
                                           i_amp=i_amp,
                                           fitting_func=None,
                                           overscan_col_start=None)
calibration500 = BaselineCalibration(path_to_cal_image_dir=path_to_cal500,
                                           Nsamp=500,
                                           centers=centers,
                                           cal1_file_strs_list=file_strs_list,
                                           i_amp=i_amp,
                                           fitting_func=None,
                                           overscan_col_start=None)

path_test_image_15 = '/home/rhur/Projects/skipperRegionalSelection/data/astroskipper_images/new/N_15/sequencer_gen-axis_1_center_10_20_width_5_5_NSAMP_15/'
path_test_image_500 = '/home/rhur/Projects/skipperRegionalSelection/data/astroskipper_images/new/N_500/sequencer_gen_axis_1_center_10_20_width_5_5_NSAMP_500/'

test_image_dict_15 = gen_image_dict(path_test_image_15, file_strs_list=['image', 'ave'])
test_image_dict_500 = gen_image_dict(path_test_image_500, file_strs_list=['image', 'ave'])  

median_test_image_15 = get_median_image(test_image_dict_15, i_amp, savefig=True, plot=False, fig_title='N15 Median Test Image', fname=plot_outdir + 'N15_median_test_image')
median_test_image_500 = get_median_image(test_image_dict_500, i_amp, savefig=True, plot=False, fig_title='N500 Median Test Image', fname= plot_outdir + 'N500_median_test_image')

data15 = median_test_image_15
data500 = median_test_image_500

dyn_var_15 = np.array([1,1,1,1,1,15,15,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
1,1,1,1,1,15,15,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
1,1,1,1,1,15,15,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,15,1,1,1,1,1,1,1,15,15,1,1,1,1,1,1,1,
1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,15,1,1,1,1,1,1,1,15,15,1,1,1,1,1,1,1,
1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,15,15,1,1,1,1,1,1,1])

dyn_var_500 = np.array([1,1,1,1,1,500,500,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
1,1,1,1,1,500,500,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
1,1,1,1,1,500,500,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,500,1,1,1,1,1,1,1,500,500,1,1,1,1,1,1,1,
1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,500,1,1,1,1,1,1,1,500,500,1,1,1,1,1,1,1,
1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,500,500,1,1,1,1,1,1,1])
NCOLCLUSTER = 100
NROWCLUSTER = 100
NROW = 600
NCOL = 3200
NSAMP = 500

#=======================================================================================================#


def PN_sequence_func(median_row, centers, fitting_func=exp_func, new_xs=None, Plot=True, Plot_fit=True):
    centers = centers[np.argwhere(np.diff(centers) > 1)[0][0]:]
    
    P500_sequence = median_row[500:700]
    P500_sequence_xs = np.arange(0, len(P500_sequence))

    if new_xs is None:
        new_xs = np.arange(0, 1000)
    
    if Plot:
        plt.scatter(P500_sequence_xs, P500_sequence,
                    color='red',
                    edgecolors="white",
                    linewidth=.7,
                    label='mean of rows')
    
    
    popt, pcov = optimize.curve_fit(f=fitting_func, xdata=P500_sequence_xs, ydata=P500_sequence, p0=(1e5, 1, 1e5))
    
    P500_sequence_fit = exp_func(new_xs, *popt)
    
    if Plot_fit:
        plt.plot(new_xs,
                 P500_sequence_fit,
                 label='exponential fit',
                 color='yellow')
        plt.title(r'$P_{multi}$ sequence', fontsize=16)
        plt.legend()
        plt.xlabel('multi sequence')
        
    return P500_sequence_fit

# we can change any lookup table after creating the BaselineCalibration object and get the corresponding baseline.
calibration15.lookup_tables['PN_sequence'] = PN_sequence_func(median_row=np.median(data15[:300], axis=0), centers=centers)
plt.savefig(fname=plot_outdir + 'N15_lookup_table_P15_sequence')
plt.show();
calibration500.lookup_tables['PN_sequence'] = PN_sequence_func(median_row=np.median(data500[:300], axis=0), centers=centers)
plt.savefig(fname=plot_outdir + 'N500_lookup_table_P500_sequence')
plt.show();

#=======================================================================================================#

baseline15 = calibration15.get_baseline(dynamic_var=dyn_var_15, NROW=NROW, NCOL=NCOL, NROWCLUSTER=NROWCLUSTER, NCOLCLUSTER=NCOLCLUSTER)
baseline500 = calibration500.get_baseline(dynamic_var=dyn_var_500, NROW=NROW, NCOL=NCOL, NROWCLUSTER=NROWCLUSTER, NCOLCLUSTER=NCOLCLUSTER)

calibration15.plot_lookup_tables(fname_prefix=plot_outdir + 'N15')
calibration500.plot_lookup_tables(fname_prefix=plot_outdir + 'N500')

plt.imshow(baseline15,
           aspect='auto',
           origin='upper',
           vmin=10000,
           vmax=30000)
plt.title('N=15 Baseline')
plt.colorbar();
plt.savefig(plot_outdir + 'N15_test_image_baseline')
plt.show();


plt.imshow((data15-baseline15),
           aspect='auto',
           origin='upper',
           vmin=-2000,
           vmax=8000)
plt.title('Residuals, N=15')
plt.colorbar()
plt.savefig(plot_outdir + 'N15_test_image_residuals')
plt.show();

plt.plot(data15[300], label='data')
plt.plot(baseline15[300], label='baseline')
plt.legend()
plt.savefig(fname=plot_outdir + 'N15_test_image_median_row_baseline')
plt.title('N15')
plt.show();

plt.plot((data15[300]-baseline15[300]), label='residuals')
plt.title('Median row, residuals, N=15')
plt.legend()
plt.ylim(-2000,8000)
plt.savefig(fname=plot_outdir + 'N15_test_image_median_row_residuals')
plt.show();

plt.imshow(baseline500,
           aspect='auto',
           origin='upper',
           vmin=10000,
           vmax=30000)
plt.title('N=500 Baseline')
plt.colorbar();
plt.savefig(plot_outdir + 'N500_test_image_baseline')
plt.show();


plt.imshow((data500-baseline500),
           aspect='auto',
           origin='upper',
           vmin=-2000,
           vmax=3000)
plt.title('Residuals, N=500')
plt.colorbar()
plt.savefig(plot_outdir + 'N500_test_image_residuals')
plt.show();

plt.plot(data500[300], label='data')
plt.plot(baseline500[300], label='baseline')
# plt.plot((baseline500-data500)[300], label='residuals')
# plt.ylim(-2000,3000)
plt.legend()
plt.title('N500')
plt.savefig(fname=plot_outdir + 'N15_test_image_median_row_baseline')
plt.show();

plt.plot((data500[300]-baseline500[300]), label='residuals')
plt.title('Median row, residuals, N=500')
plt.ylim(-2000,3000)
plt.legend()
plt.savefig(fname=plot_outdir + 'N500_test_image_median_row_residuals')
plt.show();
