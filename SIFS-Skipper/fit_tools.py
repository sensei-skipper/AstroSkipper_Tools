#!/usr/bin/env python
"""
Load and process Skipper CCD images with Regions of Interest (ROI)

"""

__all__ = ['SkipperImageROI']

import os
import numpy as np
from astropy.io import fits
from astropy.stats import sigma_clip
import re
import sys
import logging 
from fit_tools import *
logging.basicConfig()
logging.root.setLevel(logging.NOTSET)
log = logging.getLogger("SkipperImageROI")
import copy

# Utility functions

def getKeyword(header, keyword, default_value):
    """Retrieve keyword from FITS file header.

    Parameters
    ----------
    header : `astropy.io.fits.header.Header`
       FITS file header
    keyword : `str`
        FITS file header keyword
    default_value : `float`
        Default value to assign to the keyword,
        if not found in the header
    
    Returns
    -------
        The value of the requested FITS header keyword if present,
        otherwise `default_value`.
    """
    try:
        return header[keyword]  
    except RuntimeError:
        logging.basicConfig()
        log.warning('Keyword "' + keyword + '" not found, using default value ' + str(default_value))

    return default_value

def modify_string(input_str, val_index, new_val):
    """
    Modify a FITS-section-like string at a specified index.

    Parameters
    ----------
    input_str : `str`
        A section string of the form "[x:y, x:y]" found in 
        FITS header keywords describing region sections.
    val_index : `int`
        The index of the part in the string to modify. This index 
        refers to the list of split parts after removing square 
        brackets and splitting by ':' or ','.
    new_val : `str`
        The new value to insert at the specified `val_index`.

    Returns
    -------
    str
        The modified string in the format "[x:y, x:y]" with the 
        specified value replaced at `val_index`.

    Notes
    -----
    The function first removes the outer brackets from `input_str`,
    then splits by any colon or comma to obtain a list of values.
    The element at `val_index` is updated to `new_val`, and then 
    the string is rebuilt into the format `"[x:y, x:y]"`.
    """

    parts = input_str[1:-1].split(':')
    parts = re.split(r'[:,]',input_str[1:-1])
    parts[val_index] = new_val
    modified_str = '[' + ':'.join(parts[:2]) + ', ' + ':'.join(parts[2:]) + ']'
    
    return modified_str

def allAmpValues(filename,keyword,amp_offset):
    """
    Retrieve the value of a specified FITS header keyword from each amplifier.

    Parameters
    ----------
    filename : `str`
        Path to the FITS file to open.
    keyword : `str`
        Name of the header keyword to retrieve from each amplifier HDU.
    amp_offset : `int`
        The number of HDUs in the FITS file before the first amplifier HDU.
        For example, if the file has one primary HDU and amplifiers start
        at HDU 1, then `amp_offset` should be 1.

    Returns
    -------
    list
        A list of values corresponding to the specified keyword from each
        amplifier HDU in the file. If the keyword is not found in a given
        HDU, `None` is appended instead.
    """
    values=list()
    with fits.open(filename) as hdul:
        for i in range(len(hdul)-amp_offset):
            header=hdul[i+amp_offset].header
            if keyword in header:
                value=header[keyword]
                values.append(value)
            else:
                value=None
                values.append(value)
    return values

def get_input_slices(num_amps,type:str):
    """
    Prompt the user to define pixel slices for each amplifier interactively.

    Parameters
    ----------
    num_amps : `int`
        The number of amplifiers in the image.
    type : `str`
        A descriptive string (e.g., "single-sample overscan", "multi-sample overscan")
        to include in the interactive prompt for clarity.

    Returns
    -------
    list
        A list of slice objects. Each slice corresponds to the 
        (start, end) pixel indices for one amplifier. If the user 
        chooses "ALL" during the input process, the same slice is 
        replicated for the remaining amplifiers.

    Notes
    -----
    This function is interactive. It will:
    
    1. Prompt the user for the starting and ending pixel indices for each amplifier.
    2. Ask if the entered slice should apply to all subsequent amplifiers if this 
       is not the last amplifier. 
    3. If the user types "ALL", the same slice is cloned for the remaining amplifiers.

    The function logs a message via `log.info` indicating that slices are 
    being created for each amplifier. It returns as soon as enough slices 
    are gathered or when the user decides to clone the slice for the 
    remaining amplifiers. 
    """
    all_slices = list()
    log.info("Slices are created for each amp in the image")
    for amp in range(num_amps):
        start=int(input(f"Enter starting pixel for amp {amp} {type}: "))
        end=int(input(f"Enter ending pixel for amp {amp} {type}: "))
        current_slice = slice(start,end)

        if amp==amp+1:
            all_slices.append(current_slice)
        else:
            choice = input("Does the previous slice apply to remaining amps? Enter 'ALL' to clone, press Enter to continue inputing slices: " )
            if choice.upper() == "ALL":
                all_slices.extend([current_slice] * (num_amps - amp))
                break
            else:
                all_slices.append(current_slice)
    
    return all_slices

def get_slices_multiple(num_amps):
    """
    Prompt the user to define multiple Regions of Interest (ROI) for each amplifier.

    Parameters
    ----------
    num_amps : `int`
        The number of amplifiers in the image.

    Returns
    -------
    list of list of slice
        A nested list where each element corresponds to one amplifier and 
        contains a list of `slice` objects representing multiple ROI ranges.

    Notes
    -----
    This function is interactive. It will:
    
    1. Prompt the user for consecutive (start, end) pixel indices until 
       they input `0` for both start and end, which stops data collection 
       for the current amplifier.
    2. Store each pair of (start, end) in a list of `slice` objects for 
       that amplifier.
    3. If the user decides to apply the same list of ROI slices to all 
       remaining amplifiers (by entering "ALL"), the function copies 
       the first amplifier’s list of slices for all subsequent amplifiers.
    """
    all_slices = []
    for amp in range(num_amps):
        slices_for_amp = []
        while True:
            start = int(input(f"Enter starting pixel for amp {amp} ROI(s) (enter 0 to stop) or keep entering slices for multiple ROIs: "))
            end = int(input(f"Enter ending pixel for amp {amp} ROI(s) (enter 0 to stop) or keep entering slices for multiple ROIs: "))
            
            if start == 0 and end == 0:
                break
            
            slices_for_amp.append(slice(start, end))
        
        all_slices.append(slices_for_amp)
        
        if amp == 0:
            choice = input("Does the previous list of slices apply to remaining sections? Enter 'ALL' to clone, press Enter to continue: ")
            if choice.upper() == "ALL":
                all_slices.extend([slices_for_amp.copy()] * (num_amps - 1))
                break
    
    return all_slices

class SkipperImageROI:
    """
    A class to load, process, and handle Regions of Interest (ROI) 
    for Skipper CCD images.

    The class provides functionality to:
    
    - Load FITS images from a file.
    - Parse and handle various header information, such as overscan correction,
      number of samples, and CCD dimensions.
    - Apply different overscan correction methods, including serial 
      and parallel overscan subtraction.
    - Optionally define or manually input Regions of Interest (ROI) to focus 
      on specific image regions.
    - Perform gain correction if an appropriate gain file is provided.
    - Work in both interactive/manual modes and within an automated data pipeline.

    Parameters
    ----------
    filename : `str`, optional
        Path to the FITS file to open and process. If None, a filename 
        must be provided before loading data.
    ignore_samples : `int`, optional
        Number of samples at the beginning of the readout to ignore. 
        Defaults to 0.
    skip_rows : `int`, optional
        Number of leading rows to skip in the CCD readout. Defaults to 0.
    n_prescan : `int`, optional
        Number of prescan columns to ignore during processing. Defaults to 7.
    extra_overscan_margin : `int`, optional
        Additional margin (in pixels) to exclude from overscan calculations. Defaults to 4.
    ccd_n_col : `int`, optional
        Number of columns in the CCD. Defaults to 3400.
    ccd_n_row : `int`, optional
        Number of rows in the CCD. Defaults to 600.
    n_samples : `int`, optional
        Number of skipper samples per pixel. Defaults to 1.
    overscan_correction_type : `str`, optional
        Overscan correction method. Acceptable values include 'S', 'P', 'SP', or 'None'.
        Defaults to None.
    correction_method : `str`, optional
        The technique to use when performing overscan correction. 
        Common options include "MEDIAN_PER_ROW", "CUBIC_SPLINE", 
        "NATURAL_CUBIC_SPLINE", or "GAUSSIAN_FIT". Defaults to "CUBIC_SPLINE".
    ROI : `bool`, optional
        Whether or not to use Regions of Interest during processing. Defaults to True.
    manual_input_slices : `bool`, optional
        If True, prompts the user to define ROI slices interactively. Defaults to False.
    data_pipeline : `bool`, optional
        If True, the class will attempt to automate some ROI assignments based 
        on header keywords for batch processing. Defaults to False.
    calibration_single_sample : `bool`, optional
        If True, processes the image as a single-sample calibration product, 
        restricting certain corrections (e.g., only serial overscan or none). 
        Defaults to False.
    gain_correction : `bool`, optional
        If True, applies gain correction using a provided or default gain file. 
        Defaults to False.
    sifs_skipper_gain : `str`, optional
        Path to the gain file (in NumPy .npy format). If None, the class attempts 
        to load a default file named 'sifs_astroskipper_gain.npy' from the current 
        directory. Defaults to None.

    Attributes
    ----------
    filename : `str` or None
        The path to the FITS file if provided, otherwise None.
    image_processed : `bool`
        A flag indicating if the image has already been processed.
    gain : `numpy.ndarray` or None
        The gain array loaded from the specified file or None if no 
        gain correction is used or if the file cannot be loaded.

    Notes
    -----
    - After instantiating this class with desired parameters, call 
      `processImage()` to load and correct the image data.
    - The class also provides methods like `get_full_image()`, `get_image_amp()`, 
      and `save()` to retrieve, inspect, and save processed images.
    - When `manual_input_slices` is True, you will be prompted to define the ROI 
      slices interactively for each amplifier.
    - For pipeline usage (`data_pipeline=True`), the class attempts to read ROI 
      sections and overscan slices directly from the FITS headers.

    Examples
    --------
     skipper = SkipperImageROI(filename='my_skipper_image.fits', ROI=True,
                              overscan_correction_type='SP',
                            correction_method='CUBIC_SPLINE')
     skipper.processImage()
    corrected_image_data = skipper.get_full_image()
    skipper.save('my_skipper_image_corrected.fits')
    """
    def __init__(self, filename=None, ignore_samples=0, skip_rows=0,
                 n_prescan=7, extra_overscan_margin=4, ccd_n_col=3400, ccd_n_row=600, 
                 n_samples=1, overscan_correction_type=None,
                 correction_method="CUBIC_SPLINE",
                 ROI=True,manual_input_slices=False,
                 data_pipeline=False,
                 calibration_single_sample=False,
                 gain_correction = False,
                 sifs_skipper_gain = None):
          
          self.filename = filename
          self.ignore_samples = ignore_samples
          self.skip_rows = skip_rows
          self.n_prescan = n_prescan
          self.extra_overscan_margin = extra_overscan_margin
          self.ccd_n_col = ccd_n_col
          self.ccd_n_row = ccd_n_row
          self.n_samples = n_samples
          self.image_processed = False
          self.overscan_correction_type=overscan_correction_type
          self.correction_method = correction_method
          self.ROI = ROI
          self.manual_input_slices=manual_input_slices
          self.data_pipeline=data_pipeline
          self.calibration_single_sample = calibration_single_sample
          self.gain_correction = gain_correction
          self.sifs_skipper_gain = sifs_skipper_gain
          self.gain = None
    
    @property
    
    def header(self, amp=1):
        """Retrieve header of a given HDU/amplifier"""
        return self.fits_hdu_list[amp + 1].header
    

    
    def set_full_image(self, image_data):
        """Set the full image data directly."""
        self.image_data = image_data
        self.image_processed = True

    
    def processImage(self, filename=None):
        """Process CCD image.

        Parameters
        ----------
        filename : `str`
            FITS file name

        Notes
        -----
        Process image: 1) loads fits file (extracing information
        such as overscan starting point) and turns data into a
        data cube, subtracts overscan, and averages the number of
        samples.

        """
        if((not self.image_processed) or ((self.filename != filename) and (filename != None))):
            self.loadImage(filename)
            self.apply_corrections()
            #self.averageSamples() # Assumes the image is averaged from SIFS-Skipper Daq pipiline. We need to implement a averageSamples() function if that is not the case

        self.image_processed = True
    
    def imageParams(self, filename=None,ROI=None):
        ss_os_slices=list()
        ms_os_slices=list()
        ss_slices=list()
        ms_slices_arr = list()
        parallel_os = list()
        ms_slices_all=list()
        ROI = self.ROI
        
        if self.data_pipeline:
            amp_offset_init = 1
            ROI_sections_init = allAmpValues(filename,"ROISECS",amp_offset_init)
            loc_init= allAmpValues(filename,"AMPSEC",amp_offset_init)
            check_roi_init  = any(any(element is None for element in values) for values in [ROI_sections_init,loc_init])

            if not check_roi_init:
                self.ROI=True
                self.manual_input_slices = False #no option to enter input slices in pipieline mode 
            
            elif check_roi_init:
                self.ROI=False
                self.manual_input_slices=False
                log.warning("Cannot Define ROI(s) while running in Data Pipeline Mode: Failed to Find 'ROISECS', and 'AMPSEC' in the Header of "+str(filename))
                log.info("Will assume no ROI(s) in"+str(filename))


        if ROI:
            header_index=1
            self._amp_offset=1
        elif ".fits" in filename and not self.calibration_single_sample:
            header_index = 0
            self._amp_offset=0
        elif ".fz" in filename and not self.calibration_single_sample:
            header_index=1
            self._amp_offset=1
        elif not ROI and self.calibration_single_sample:
            header_index=1
            self._amp_offset=1
      
        self.n_samples = int(
            float(getKeyword(self.fits_hdu_list[header_index].header, "NSAMP", self.n_samples))
            )
        self.ccd_n_col = int(
                int(getKeyword(self.fits_hdu_list[1].header, "CCDNCOL", self.ccd_n_col)) / 2
            )
        self.ccd_n_row = int(
                int(getKeyword(self.fits_hdu_list[1].header, "CCDNROW", self.ccd_n_row)) / 2
            )
        self.readout_rows = int(
            float(getKeyword(self.fits_hdu_list[header_index].header, "NROW", self.n_samples))
            )
        self.readout_columns = int(
            float(getKeyword(self.fits_hdu_list[header_index].header, "NCOL", self.n_samples))
            )
        self.n_prescan = int(getKeyword(self.fits_hdu_list[header_index].header, "CCDNPRES", self.n_prescan))

        self.n_amps = len(self.fits_hdu_list) 

        if ROI:
            loc = allAmpValues(filename,"AMPSEC",self._amp_offset)
            OSSend=self.readout_columns
            ROI_sections = allAmpValues(filename,"ROISECS",self._amp_offset)
            if any(val is None for val in ROI_sections):
                test_roi = list()
                number_ROIs = allAmpValues(filename,"NROIS",self._amp_offset)

                if any(val is None for val in ROI_sections):
                    ROI_sections = [None]  
                else:                  
                    for i in range(int(number_ROIs[0])):
                        roi = allAmpValues(filename,"ROISEC{}".format(i+1),self._amp_offset)
                        test_roi.append(roi)
                    
                    ROI_sections = [' '.join(chars) for chars in zip(*test_roi)]
                    
                    if len(ROI_sections)==0:
                        ROI_sections = [None]



            Bias_sec = allAmpValues(filename,"BIASSEC",self._amp_offset)
            check_roi  = any(any(element is None for element in values) for values in [ROI_sections,loc])
            
       
            if self.manual_input_slices:
                self._amp_offset=1
                self.n_amps = int(input("Enter the number of amplifiers (HDUs) in the image: "))
                self.ss_os_slice = get_input_slices(self.n_amps,type="single-sample overscan")
                self.ms_os_slice=get_input_slices(self.n_amps,type="multi-sample overscan")
                self.ms_slices = get_slices_multiple(self.n_amps)
                self.serial_overscan_start = self.ccd_n_col #+ self.n_prescan
                self.parallel_os = self.ccd_n_row
                self.parallel_os = [self.parallel_os]*self.n_amps


                for i in range(self.n_amps):
                    ms_slice = self.ms_slices[i]
                    ss_slices.append(slice(0,ms_slice[0].start))
                    for i in range(len(ms_slice)-1):
                        ss_slices.append(slice(ms_slice[i].stop, ms_slice[i + 1].start))
                    ss_slices.append(slice(ms_slice[-1].stop, self.serial_overscan_start))
                
                self.ss_slices = np.array(ss_slices).reshape(self.n_amps,-1)

                for i in range(self.n_amps):
                    ms_slice = self.ms_slices[i]
                    ms_slice_copy = copy.deepcopy(ms_slice)
                    ms_slice_copy.append(self.ms_os_slice[i])
                    ms_slices_all.append(ms_slice_copy)
                
                self.ms_slices_all = ms_slices_all



            if not self.manual_input_slices:
                if check_roi:
                    
                    log.warning("Cannot Process Image: Failed to Find 'ROISECS', and 'AMPSEC' in the Header of "+str(filename))
                    log.info("Alternatively, set 'manual_roi_input' to 'True' to enter ROI information manually")  
                    sys.exit(1)
                else:
                    pattern = re.compile(r'\[(\d+:\d+),\d+:\d+\]')
                    amp_direction = pattern.findall(loc[0])
               
                    for i in range(self.n_amps-self._amp_offset):
                        if pattern.findall(loc[i]) == amp_direction:
                            all_OS = ROI_sections[i].split('[')[2]
                            all_OS = re.split(r'[:,]',all_OS)
                            singleOSend = int(all_OS[0])-1
                            multiOSend = all_OS[1]
                            bias_sec = Bias_sec[i]
                            s_b = modify_string(bias_sec,1,str(singleOSend))
                            m_s = modify_string(bias_sec,0,str(singleOSend))
                        elif pattern.findall(loc[i]) != amp_direction:
                             all_OS = ROI_sections[i].split('[')[1]
                             all_OS = re.split(r'[:,]',all_OS)
                             singleOSstart = int(all_OS[0])-1
                             bias_sec = Bias_sec[i]
                             singleOSend = int(re.split(r'[:,]',bias_sec[1:-1])[1]) 
                             
                             m_s = modify_string(bias_sec,0,str(singleOSstart))
                             m_s = modify_string(m_s,1,str( int(all_OS[1])))

                             s_b =  modify_string(bias_sec,0,str(int(all_OS[1])))
                             s_b = modify_string(s_b,1,str(singleOSend))


                        split_string_single_OS = s_b.split(',')  #OS_single[i].split(',')
                        split_string_multi_OS = m_s.split(',')  #OS_multi[i].split(',')

                        
                        OS_single_sections =  split_string_single_OS[0].split('[')[1].split(']')[0].strip().split(':')
                        OS_multi_sections = split_string_multi_OS[0].split('[')[1].split(']')[0].strip().split(':')
                        
                        parallel_OS_start = split_string_single_OS[1].split(']')[0].strip().split(':')
                        parallel_os.append(parallel_OS_start[(i+1) % len(parallel_OS_start)])
                        
                        ss_os_slice = slice(int(OS_single_sections[0]), int(OS_single_sections[1]))
                        ss_os_slices.append(ss_os_slice)
                        ms_os_slice = slice(int(OS_multi_sections[0]), int(OS_multi_sections[1]))
                        ms_os_slices.append(ms_os_slice)
                        
                        matches = pattern.findall(ROI_sections[i])
                        ms_slices = [slice(int(start)-1, int(end)) for start, end in (match.split(':') for match in matches)]
                        ms_slices_all.append([slice(int(start)-1, int(end)) for start, end in (match.split(':') for match in matches)])
                        num_elements = len(ms_slices)
                    
            
                        if pattern.findall(loc[i]) == amp_direction :
                            ms_slices = ms_slices[:-1]
                            ms_slices_arr.append(ms_slices)

                            ss_slices.append(slice(0, ms_slices[0].start))
                            
                            for i in range(len(ms_slices) - 1):
                                ss_slices.append(slice(ms_slices[i].stop, ms_slices[i + 1].start))
                            
                            ss_slices.append(slice(ms_slices[-1].stop,  ss_os_slice.start))
                        else:
                            ms_slices = ms_slices[1:]
                            ms_slices_arr.append(ms_slices)
                            ss_slices.append(slice(ms_os_slice.stop, ms_slices[0].start))

                            for i in range(len(ms_slices) - 1):
                                ss_slices.append(slice(ms_slices[i].stop, ms_slices[i + 1].start))
                            ss_slices.append(slice(ms_slices[-1].stop, OSSend))
                    
                    ss_slices = np.array(ss_slices).reshape(int(len(ss_slices)/num_elements),num_elements)

                    self.ss_os_slice = ss_os_slices
                    self.ms_os_slice = ms_os_slices
                    self.ms_slices = ms_slices_arr
                    self.ss_slices = ss_slices
                    self.parallel_os = parallel_os
                    self.parallel_os = [int(value) for value in self.parallel_os]
                    self.ms_slices_all = ms_slices_all
            
        elif not ROI and self.calibration_single_sample:
            serial_OS = list()
            Bias_sec = allAmpValues(filename,"BIASSEC",self._amp_offset)
            
            for i in range(self.n_amps-self._amp_offset):
                bias_sec = Bias_sec[i]
                string = bias_sec[1:-1].split(',')
                start, end = map(int, string[0].split(':'))
                serial_OS.append(slice(start,end))
            
            self.ss_os_slice = serial_OS
        
        if self.gain_correction:
            if self.sifs_skipper_gain == None:
                self.sifs_skipper_gain = 'sifs_astroskipper_gain.npy'

                if os.path.exists(self.sifs_skipper_gain):
                    self.gain = np.load(self.sifs_skipper_gain)
                
                else:
                    log.warning("Cannot load default gain file: 'sifs_astroskipper_gain.npy' gain correction will not be applied")
                    self.gain = [1.]*self.n_amps
            else:
                self.sifs_skipper_gain= self.sifs_skipper_gain
                if os.path.exists(self.sifs_skipper_gain):
                    self.gain = np.load(self.sifs_skipper_gain)
                else:
                    log.warning(f"Cannot load gain file: '{self.sifs_skipper_gain}' gain correction will not be applied")
                    self.gain = [1.]*self.n_amps
        else:
            self.gain = [1.]*self.n_amps
        

    def loadImage(self,filename=None):

        self.image_processed = False

        if(self.filename == None):
            raise ValueError('SkipperImage::loadImage -  No filename given')
        else:
            filename=self.filename

        try:
            self.fits_hdu_list = fits.open(self.filename)
        except IOError:
            raise IOError("Could not open FITS file " + str(filename))
        
        self.imageParams(filename=filename)


        self.image_data = np.zeros(shape=(self.n_amps-self._amp_offset,self.readout_rows,self.readout_columns))
        
        for i in range(self.n_amps-self._amp_offset):
            self.image_data[i]=fits.open(filename)[i+self._amp_offset].data


    
    # Image Correction Methods 
    
    def serial_overscan_correction(self, image, serial_overscan_start, serial_overscan_end,correction_method):
        correction_method = self.correction_method
        corrected_serial_ovsercan_image = np.copy(image)
        overscan_array = np.median(corrected_serial_ovsercan_image[:, serial_overscan_start:serial_overscan_end], axis=-1)

        if correction_method == "MEDIAN_PER_ROW":
            overscan_array = overscan_array
        elif correction_method == "CUBIC_SPLINE":
            overscan_array = cubic_spline_fit(overscan_array)
        
        elif correction_method =="NATURAL_CUBIC_SPLINE":
            overscan_array =  natural_cubic_spline(overscan_array)
  
        elif correction_method == "GAUSSIAN_FIT":
            overscan_array=gaussian_fit(overscan_array,corrected_serial_ovsercan_image,serial_overscan_start+self.extra_overscan_margin,None,None,serial=True)
       
        corrected_serial_ovsercan_image -= overscan_array[:, np.newaxis]

        return corrected_serial_ovsercan_image
    
    def parallel_overscan_correction(self,image,parallel_overscan_start,correction_method,multi_sample_slices=None):
        corrected_parallel_ovsercan_image = np.copy(image)
        
        if (corrected_parallel_ovsercan_image.shape[0]//parallel_overscan_start) <= 1:
            overscan_array = np.median(corrected_parallel_ovsercan_image[parallel_overscan_start+self.extra_overscan_margin:], axis=0)
            
        
        elif (corrected_parallel_ovsercan_image.shape[0]//parallel_overscan_start) >= 1:
            overscan_array = np.median(corrected_parallel_ovsercan_image[:parallel_overscan_start], axis=0)

        
        if correction_method == "MEDIAN_PER_ROW":
            overscan_array = overscan_array
        
        elif correction_method == "CUBIC_SPLINE":
            overscan_array = cubic_spline_fit(overscan_array)
        
        elif correction_method =="NATURAL_CUBIC_SPLINE":
            overscan_array =  natural_cubic_spline(overscan_array)
  
        
        elif correction_method == "GAUSSIAN_FIT":
            overscan_array= gaussian_fit(overscan_array,corrected_parallel_ovsercan_image,parallel_overscan_start,None,multi_sample_slices,serial=False)

        corrected_parallel_ovsercan_image -= overscan_array


       
        return corrected_parallel_ovsercan_image
    
    def serial_parallel_overscan_correction(self, image,serial_overscan_start,parallel_overscan_start,correction_method):
        corrected_serial_ovsercan_image = self.serial_overscan_correction(image,serial_overscan_start,correction_method)
        self.image_data = corrected_serial_ovsercan_image
        corrected_parallel_ovsercan_image = self.parallel_overscan_correction(corrected_serial_ovsercan_image,parallel_overscan_start,correction_method)
        self.image_data = corrected_parallel_ovsercan_image
    
    def serial_roi(self,image, correction_method, ss_os_slice:slice, ms_os_slice:slice, ss_slices:list, ms_slices:list):
        corrected_serial_ovsercan_image = np.copy(image)
        ss_os_array = np.median(corrected_serial_ovsercan_image[:, ss_os_slice.start+self.extra_overscan_margin:ss_os_slice.stop], axis=-1)
        ms_os_array = np.median(corrected_serial_ovsercan_image[:, ms_os_slice.start+self.extra_overscan_margin:ms_os_slice.stop], axis=-1)
     
      
        
        if correction_method=="MEDIAN_PER_ROW":
            ss_os_array = ss_os_array
            ms_os_array = ms_os_array
        
        elif correction_method =="CUBIC_SPLINE":
            ss_os_array =  cubic_spline_fit(ss_os_array)
       
            ms_os_array = cubic_spline_fit(ms_os_array)
        
        elif correction_method =="NATURAL_CUBIC_SPLINE":
            ss_os_array =  natural_cubic_spline(ss_os_array)

            ms_os_array = natural_cubic_spline(ms_os_array)


        elif correction_method == "GAUSSIAN_FIT":
            ss_os_array=  cubic_spline_fit(ss_os_array) 
            ms_os_array= gaussian_fit(ms_os_array,corrected_serial_ovsercan_image,ms_os_slice.start+self.extra_overscan_margin,ms_os_slice.stop,None,serial=True)

        for ss_slice in ss_slices:
      
            corrected_serial_ovsercan_image[:, ss_slice] -= ss_os_array[:, np.newaxis]
        
        for ms_slice in ms_slices:
            corrected_serial_ovsercan_image[:, ms_slice] -= ms_os_array[:, np.newaxis]
        
        
        corrected_serial_ovsercan_image[:, ss_os_slice] -= ss_os_array[:, np.newaxis]
        corrected_serial_ovsercan_image[:, ms_os_slice] -= ms_os_array[:, np.newaxis]


        
        return corrected_serial_ovsercan_image
 
    def parallel_roi(self,image,parallel_overscan_start,correction_method,multi_sample_slices=None):
        corrected_parallel_ovsercan_image = self.parallel_overscan_correction(image,parallel_overscan_start,correction_method,multi_sample_slices=multi_sample_slices)

        return corrected_parallel_ovsercan_image

    def serial_parallel_roi(self,image,parallel_overscan_start,correction_method, ss_os_slice:slice, ms_os_slice:slice, ss_slices:list, ms_slices:list,multi_sample_slices):
        corrected_serial_ovsercan_image = self.serial_roi(image,correction_method, ss_os_slice, ms_os_slice, ss_slices, ms_slices)
        
        corrected_parallel_ovsercan_image = self.parallel_roi(corrected_serial_ovsercan_image,parallel_overscan_start,correction_method,multi_sample_slices=multi_sample_slices)

        return corrected_parallel_ovsercan_image
    
    def apply_corrections(self, image=None, overscan_correction_type=None,correction_method=None,ROI=None):
        image = self.image_data
        overscan_correction_type = self.overscan_correction_type
        correction_method=self.correction_method
        ROI = self.ROI
        corrected_image = np.zeros_like(image)
        overscan_correction_type=overscan_correction_type
        if overscan_correction_type is None or overscan_correction_type.lower() not in ['s','p','sp','none']:
            logging.warning("You must select an overscan correction type: -S (serial overscan subtraction), -P (parallel overscan subtraction), -SP (serial and parallel overscan subtraction)")
            sys.exit(1)
        if overscan_correction_type.lower() == 'none':
            for i in range(self.n_amps-self._amp_offset):
                corrected_image[i] = image[i]
       
        if ROI:
            overscan_correction_type=overscan_correction_type.lower()
            if overscan_correction_type == 's':
                for i in range(self.n_amps-self._amp_offset):
                   corrected_image[i] = self.serial_roi(image[i],correction_method,self.ss_os_slice[i],self.ms_os_slice[i],self.ss_slices[i],self.ms_slices[i])
            
            if overscan_correction_type =='p':
                for i in range(self.n_amps-self._amp_offset):
                    corrected_image[i] = self.parallel_roi(image[i],self.parallel_os[i],correction_method,multi_sample_slices=self.ms_slices_all[i])
            
            self.image_data=corrected_image

            if overscan_correction_type == 'sp':
                for i in range(self.n_amps-self._amp_offset):
  
                    corrected_image[i] = self.serial_parallel_roi(image[i],self.parallel_os[i],correction_method,self.ss_os_slice[i],self.ms_os_slice[i],self.ss_slices[i],self.ms_slices[i],multi_sample_slices=self.ms_slices_all[i])
            
            self.image_data=corrected_image
                
        if not ROI and self.calibration_single_sample:
            overscan_correction_type=overscan_correction_type.lower()
            if overscan_correction_type == 's':
                for i in range(self.n_amps-self._amp_offset):
                    corrected_image[i] = self.serial_overscan_correction(image[i],self.ss_os_slice[i].start, self.ss_os_slice[i].stop,correction_method)
            
            elif overscan_correction_type == 'none':
                 for i in range(self.n_amps-self._amp_offset):
                     corrected_image[i] = image[i]
            else:
                logging.warning("For single sample calibration products only -S or -None (serial overscan subtraction) are supported")
                sys.exit(1)
            self.image_data = corrected_image

    def get_full_image(self):
        return self.image_data
            
    def get_image_amp(self,amp=0):
        return self.image_data[amp]
    
    def save(self, filename=None,injected=True):
        if injected:
            self.processImage()
        
        new_hdul = fits.HDUList()

        if filename is None:
            fname = os.getcwd() + "/"+ self.filename.split("/")[-1].split(".")[0] + "_processed.fits"
        else:
            fname = filename 
        
        hdulist =  self.fits_hdu_list

        hdu_count = len(hdulist)

        if self.image_data.shape[0] != hdu_count:
             new_hdul.append(fits.PrimaryHDU(header=hdulist[0].header))
        for i in range(hdu_count-self._amp_offset):
            header = hdulist[i+self._amp_offset].header 
            new_hdu = fits.PrimaryHDU(self.image_data[i]/self.gain[i])
            new_hdu.header.update(header)
            new_hdul.append(new_hdu)
          
        new_hdul.writeto(fname, overwrite=True)
    
