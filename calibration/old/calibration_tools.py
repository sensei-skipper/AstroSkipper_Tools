import numpy as np
import os
import matplotlib.pyplot as plt
from astropy.io import fits
from scipy import interpolate
from scipy import optimize

# utility fxns

def exp_func(xs, A, C, D):
    return A*np.exp(-C*xs)+D

def gen_image_dict(path, file_strs_list):
    image_dict = {}
    filenames = os.listdir(path=path)
    filenames.sort
    for filename in filenames:
        if all(string in filename for string in file_strs_list):
            name = path + filename
            image_dict[filename[:]] = fits.open(name=name)
    if image_dict:
        return image_dict
    else:
        print("Did not find files that match strings in file_strs_list")
    return image_dict

def get_median_image(image_dict, i_amp, origin='upper', aspect='auto', plot=True, savefig=False, fname='./astroskipper_calibration_tools/median_image', fig_title='Median Image'):
    median_image = np.median(np.array([image_dict[key][i_amp].data for key in image_dict]), axis=0)
    plt.imshow(median_image, origin=origin, aspect=aspect)
    plt.colorbar();
    plt.title(fig_title)
    if savefig:
        plt.savefig(fname=fname)
    if not plot:
        plt.close()
    else:
        plt.show();
    return median_image


class BaselineCalibration():
    '''
    Object which calculates perturbations to baseline (vertical shifts not yet included)
    Can get baseline for arbitrary image using self.get_baseline_val()
    '''
    def __init__(self, path_to_cal_image_dir, Nsamp, centers, cal1_file_strs_list=None, i_amp=0, fitting_func=None, overscan_col_start=None, p0_PNdN=None, p0_P1dN=None, p0_PNsequence=None, p0_PNdnu=None, p0_P1dnu=None, new_xs=None):
        self.path_to_cal_image_dir = path_to_cal_image_dir
        self.Nsamp = Nsamp
        self.centers = centers
        self.i_amp = i_amp
        self.overscan_col_start = overscan_col_start
        if cal1_file_strs_list == None:
            print(f"Did not enter file_strs. Using 'ave' and '{self.Nsamp}'")
            self.cal1_file_strs_list = ['ave', str(self.Nsamp)]
        else:
            self.cal1_file_strs_list = cal1_file_strs_list
        self.cal1_image_dict = gen_image_dict(path=self.path_to_cal_image_dir, file_strs_list=self.cal1_file_strs_list)
        self.median_cal1_imag = get_median_image(image_dict=self.cal1_image_dict, i_amp=self.i_amp, plot=False)
        self.median_cal_row = np.median(self.median_cal1_imag, axis=0)
        if fitting_func is None:
            self.fitting_func = exp_func
        else:
            self.fitting_func = fitting_func
        if new_xs is None:
            new_xs = np.arange(0, 5000)
        p0s = [p0_P1dN, p0_PNdN, p0_PNsequence, p0_P1dnu, p0_PNdnu]
        new_p0s = [p0 if p0 is not None else (1e5, 1e-2, 1e5) for p0 in p0s]
        self.p0_P1dN, self.p0_PNdN, self.p0_PNsequence, self.p0_P1dnu, self.p0_PNdnu = new_p0s
        self.lookup_tables = self._gen_lookup_tables(new_xs=new_xs)

    def plot_median_cal1_image(self, origin='upper', aspect='auto'):
        self.plot_image(ys=self.median_cal1_image, origin=origin, aspect=aspect)
        plt.title('Median Calibration Image 1' + f'{self.Nsamp}')
        plt.show();
    
    def plot_median_cal2_image(self, origin='upper', aspect='auto'):
        self.plot_image(ys=self.plot_median_cal2_image, origin=origin, aspect=aspect)
        plt.title('Median Calibration Image 2' + f'{self.Nsamp}')
        plt.show();

    def plot_image(self, image, origin='upper', aspect='auto'):
        plt.imshow(image,
                   origin=origin,
                   aspect=aspect)
        plt.colorbar()

    def plot_calibration1_images(self, origin='upper', aspect='auto'):
        for key in self.cal1_image_dict:
            plt.imshow(self.cal1_image_dict[key][self.i_amp].data,
                   origin=origin,
                   aspect=aspect)
            plt.title(key, fontsize=16)
            plt.colorbar()
            plt.show();

    def _gen_lookup_tables(self, new_xs):
        lookup_tables = dict()
        lookup_tables["PN_dN"] = self._gen_lookup_table(y="PN", x_str="dN", new_xs=new_xs)
        lookup_tables["P1_dN"] = self._gen_lookup_table(y="P1", x_str="dN", new_xs=new_xs)
        lookup_tables["PN_dnu"] = self._gen_lookup_table(y="PN", x_str="dnu", new_xs=new_xs)
        lookup_tables["P1_dnu"] = self._gen_lookup_table(y="P1", x_str="dnu", new_xs=new_xs)
        lookup_tables["PN_sequence"] = self._gen_lookup_table(y="PN", x_str="sequence", new_xs=new_xs)
        return lookup_tables

    def _gen_lookup_table(self, y, x_str, new_xs): 
        centers = self.centers[np.argwhere(np.diff(self.centers) > 1)[0][0]:] # set centers var to start where calibration image 1 pattern starts
        if self.overscan_col_start is not None: 
            last_center_index = np.argwhere(centers > self.overscan_col_start)[0][0] # set centers var to stop before overscan
            centers = centers[:last_center_index] 
        if x_str == 'dN':
            if y=="P1":
                P1_centers = np.array(centers-1)[1:]
                xs = np.diff(centers)-1
                ys = self.median_cal_row[P1_centers]
                self.raw_P1_dN_xs = xs
                self.raw_P1_dN_ys = ys
                p0 = self.p0_P1dN
            elif y=="PN":
                ys = self.median_cal_row[centers][1:] 
                xs = np.diff(centers)
                self.raw_PN_dN_ys = ys
                self.raw_PN_dN_xs = xs
                p0 = self.p0_PNdN
        elif x_str == 'dnu':
            #TODO:
            print("Still need to implement vertical clock effect!")
            p0 = self.p0_P1dnu
            p0 = self.p0_PNdnu
            return None
        elif y == 'PN' and x_str=='sequence':
            ys = self.median_cal_row[self.centers[0]+1:centers[0]+1]
            if len(ys) == 0:
                return None
            xs = np.arange(0, len(ys))
            self.raw_PN_sequence_ys = ys
            self.raw_PN_sequence_xs = xs
            p0 = self.p0_PNsequence

        popt, pcov = optimize.curve_fit(f=self.fitting_func,
                                        xdata=xs,
                                        ydata=ys,
                                        p0=p0)
        
        # save popt vals for reproducibility
        if y=='P1':
            if x_str=='dN':
                self.popt_P1dN = popt
            elif x_str=='dnu':
                self.popt_P1dnu = popt
        elif y=='PN':
            if x_str=='dN':
                self.popt_PNdN = popt
            elif x_str=='dnu':
                self.popt_PNdnu = popt
            elif x_str=='sequence':
                self.popt_PNsequence = popt
        ys_fit = self.fitting_func(new_xs, *popt)        
        return ys_fit
    
    def plot_lookup_tables(self, plot_fit=True, savefig=True, fname_prefix=None):
        if (savefig) and (fname_prefix is None):
            print("Missing arg 'fname'. Input fname to save plot.")
            return
        plt.scatter(self.raw_P1_dN_xs, self.raw_P1_dN_ys,
                    edgecolors='white',
                    color='green',
                    label='raw')    
        if plot_fit:
            plt.plot(self.lookup_tables['P1_dN'],
                label='exponential fit',
                color='orange')
        plt.title(fr'$P_{1}[d_{{{self.Nsamp}}}]$', fontsize=16)
        plt.legend()
        plt.xlabel(fr'd{{{self.Nsamp}}}')
        if savefig:
          plt.savefig(fname=fname_prefix + '_lookup_table_'+f'P1d{self.Nsamp}')
        plt.show()

        plt.scatter(self.raw_PN_dN_xs, self.raw_PN_dN_ys,
                edgecolors='white',
                color='red',
                label='raw')    
        if plot_fit:
            plt.plot(self.lookup_tables['PN_dN'],
                label='exponential fit',
                color='orange')
        plt.title(fr'$P_{{{self.Nsamp}}}[d_{{{self.Nsamp}}}]$', fontsize=16)
        plt.legend()
        plt.xlabel(fr'd{{{self.Nsamp}}}')
        if savefig:
            plt.savefig(fname=fname_prefix + '_lookup_table_'+ f'P{self.Nsamp}d{self.Nsamp}')
        plt.show();
        if hasattr(self, 'raw_PN_sequence_ys'):
            plt.scatter(self.raw_PN_sequence_xs, self.raw_PN_sequence_ys,
                        edgecolors='white',
                        color='purple',
                        label='raw')    
            if plot_fit:
                plt.plot(self.lookup_tables['PN_sequence'],
                    label='exponential fit',
                    color='orange')
            plt.title(fr'$P_{{{self.Nsamp}}}[sequence]$', fontsize=16)
            plt.legend()
            plt.xlabel(fr'sequence')
            if savefig:
                plt.savefig(fname=fname_prefix + '_lookup_table_' + f'P{self.Nsamp}sequence')
            plt.show();

    def get_expanded_dynamic_var(self, dynamic_var, NROW, NCOL, NROWCLUSTER, NCOLCLUSTER):
        dynamic_var_reshaped = np.reshape(dynamic_var, newshape=(NROW//NROWCLUSTER, NCOL//NCOLCLUSTER))
        dynamic_var_expanded = np.ones(shape=(dynamic_var_reshaped.shape[0]*NROWCLUSTER, dynamic_var_reshaped.shape[1]*NCOLCLUSTER), dtype=np.int64)
        for i, row in enumerate(dynamic_var_reshaped):
            for j, val in enumerate(row):
                if val == self.Nsamp:
                    dynamic_var_expanded[i*NROWCLUSTER:NROWCLUSTER*(i+1), j*NCOLCLUSTER:NCOLCLUSTER*(j+1)] = self.Nsamp
        return dynamic_var_expanded

    def _get_dN_array(self, dynamic_var_expanded):
        dN_array = np.zeros_like(dynamic_var_expanded.flatten(), dtype=np.int64)
        distance = 0 # distance of 0 means no contribution from dN
        for i, element in enumerate(dynamic_var_expanded.flatten()):
            dN_array[i] = distance
            if element > 1:
                distance = 1
            elif element == 1 and distance != 0:
                distance += 1
        dN_array = dN_array.reshape((dynamic_var_expanded.shape[0], dynamic_var_expanded.shape[1]))
        return dN_array
    
    def get_baseline_val(self, i_col_cur, N_cur, dN_cur, P1_dnu, P1_dN, PN_dN, PN_dnu, PN_SS, P1_SS, PN_sequence, delta_V=5, sequence_cur=None):
        '''
        Gets baseline value of one pixel.
        As is, function is essentially just subtracting P1_d500 or P500_d500 given nsamp of current pixel
        And then, if there is a sequence of 
        '''
        baseline_val = 0
        if N_cur == 1: # if current pixel has # samples N=1, reassign PN_* arrays
    #         P_N_d_nu = P_1_d_nu
            PN_dN = P1_dN
            PN_SS = P1_SS
        elif N_cur < 1:
            return "N_cur must be greater or equal to 1"
    #     if dN_cur >= i_col_cur:  # if most recent multi sample is in previous row, need vertical clock shift
    #         baseline = P_N_d_N[d_N + delta_V] + P_N_d_nu[i_col_cur] - P_N_SS
    #         baseline_val += PN_dN[dN_cur] # ignore it for now
    #     elif dN_cur < i_col_cur:
    #         print('case 2')
        in_N_sequence_region = (N_cur > 1) and (PN_sequence is not None)
        if in_N_sequence_region:  # use only PN_sequence if exists
            sequence_cur += 1
            baseline_val += PN_sequence[sequence_cur]
        elif N_cur == 1 and sequence_cur > -1: # restart sequence_cur if not in Nsequence region
            sequence_cur = -1
        if not in_N_sequence_region:
            if i_col_cur >= len(PN_dN):
                baseline_val += PN_dN[-1]
            else:
                baseline_val += PN_dN[dN_cur]
        return baseline_val, sequence_cur
    
    def get_baseline(self, dynamic_var, NROW, NCOL, NROWCLUSTER, NCOLCLUSTER):
        dynamic_var_expanded = self.get_expanded_dynamic_var(dynamic_var, NROW, NCOL, NROWCLUSTER, NCOLCLUSTER)
        dN_array = self._get_dN_array(dynamic_var_expanded)

        baseline = np.zeros_like(dynamic_var_expanded, dtype=np.float64)
        for i, row in enumerate(baseline):
            sequence_cur = -1
            for j, pixel_val in enumerate(row):
                baseline_val, sequence_cur = self.get_baseline_val(i_col_cur=j,
                                                N_cur=dynamic_var_expanded[i, j],
                                                dN_cur=dN_array[i, j],
                                                P1_dnu=self.lookup_tables["P1_dnu"],
                                                P1_dN=self.lookup_tables["P1_dN"],
                                                PN_dN=self.lookup_tables["PN_dN"],
                                                PN_dnu=self.lookup_tables["PN_dnu"],
                                                PN_SS=0,
                                                P1_SS=0,
                                                PN_sequence=self.lookup_tables["PN_sequence"],
                                                delta_V=5,
                                                sequence_cur=sequence_cur)
                baseline[i, j] = baseline_val

        return baseline
