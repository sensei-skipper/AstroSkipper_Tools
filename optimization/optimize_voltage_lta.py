import numpy as np
from pathlib2 import Path
from subprocess import call
import os, sys, time, datetime

sys.path.append('path/to/device/device')
from lta import Lta


def take_image(exptime=0.0,nrow=None,nsamp=None):
    """ Peforms readout 

    Parameters:
    -----------
    exptime : exposure time (seconds)
    nrow : number of rows to read ('lta NROW')
    nsamp : number of samples to read ('lta NSAMP')

    Returns:
    --------
    None
    """

    Lta().read(nrow=nrow,nsamp=nsamp)
    time.sleep(1)

def make_dir():
    """ Makes directory to save images 

    Parameters:
    -----------
    None

    Returns:
    --------
    direcotry path 
    """

    today = datetime.datetime.now().strftime("%Y%m%d")
    OutDir = './images/%s_Voltage_Optimization/'%today
    if not os.path.exists(OutDir): os.makedirs(OutDir)

    return OutDir

def source_file(file):
    """ runs bash scripts 

    Parameters:
    -----------
    file: name of the ".sh" file/path to the file 

    Returns:
    --------
    None 
    """
    call('bash'+ " " + file, shell=True)


def get_defaults():
    """ defines the default names and voltage values in the voltage_file. Names with "_" indicate low voltage values while vames wihtout "_"
        indicate high voltage values 

    Parameters:
    -----------
    None

    Returns:
    --------
    Dictionary of the voltages names and corresponding default values 
    """

    return {"v_":-2.75, "v":-0.5, "t_":-3.5, "t":-1,"h_":-4.5,"h":-2,"s_":-10, "s":-3,"o_":-8, "o":-4,"r_":-4, "r":7,"d_":-10,"d":-1}    

def scan(voltage_file,voltage_name,nrow,nsamp):
    """ Peforms voltage scan of a single pair of low and high voltage values. 
        All other volatges values are set to default while performing a grid scan of the low/high 
        values for the selected voltage name 

    Parameters:
    -----------
    voltage_file : volatge file or path to volatge file
    voltage_name: volatge name of the desired volatge to scan through 
    nsamp : number of samples to read ('lta NSAMP')
    nrow:  number of rows to read ('lta nrow')

    Returns:
    --------
    None
    """
    dv=-0.1
    del_v = 1.5
    key_words = get_defaults()

    voltage_file = Path(voltage_file)
    voltage_text = voltage_file.read_text()

    if voltage_name not in list(key_words.keys()):
        print("Unknown Voltage Name")
        exit()
    else: 
        low_range = np.around(np.arange(key_words[voltage_name+"_"]+del_v,key_words[voltage_name+"_"]-del_v,dv), decimals=2)
        high_range = np.around(np.arange(key_words[voltage_name]+del_v,key_words[voltage_name]-del_v,dv),decimals=2)


        for key, values in key_words.items():
            voltage_text = voltage_text.replace(key,str(values[0][0]),1)
            voltage_file.write_text(voltage_text)

        
        voltage_text = voltage_text.replace(str(key_words[voltage_name+"_"]),voltage_name+"_",1)
        voltage_file.write_text(voltage_text)

        voltage_text = voltage_text.replace(str(key_words[voltage_name]),voltage_name,1)
        voltage_file.write_text(voltage_text)

        for v_low in low_range:
             voltage_text = voltage_text.replace(key_words[voltage_name+"_"],str(v_low),1)
             voltage_file.write_text(voltage_text)
             
             for v_high in high_range:
                 voltage_text = voltage_text.replace(key_words[voltage_name], str(v_high),1)
                 voltage_file.write_text(voltage_text)
                 img_name = make_dir + "_"+ voltage_name +"_" + str(v_high) + '_' + voltage_name  +'_' + str(v_low) + '_'
                 Lta.talk("name",img_name)
                 source_file(voltage_file)
                 take_image(nrow=nrow, nsamp=nsamp)
                 voltage_text = voltage_text.replace(str(v_high),voltage_name,1)
                 voltage_file.write_text(voltage_text)
            
             voltage_text = voltage_text.replace(str(v_low),voltage_name+"_",1)
             voltage_file.write_text(voltage_text)

def scan_all(voltage_file, nrow, nsamp):
    """ Runs the scan for all volatge names
    Parameters:
    -----------
    voltage_file : volatge file or path to volatge file
    nsamp : number of samples to read ('lta NSAMP')
    nrow:  number of rows to read ('lta nrow')

    Returns:
    --------
    None
    """
    names = list(get_defaults().keys())[1::2]
    for name in names:
        scan(voltage_file,name,nrow,nsamp)


def run_scan(voltage_file, voltage_name,nrow,nsamp,run_all=False):

    """ Runs the voltage scan

    Parameters:
    -----------
    voltage_file : volatge file or path to volatge file
    voltage_name: volatge name of the desired volatge to scan through 
    nsamp : number of samples to read ('lta NSAMP')
    nrow:  number of rows to read ('lta nrow')
    run_all: indcates if the scan will be performed over all voltages 

    Returns:
    --------
    None
    """

    if run_all:
        voltage_name=None
        scan_all(voltage_file, nrow, nsamp)
    else:
        scan(voltage_file, voltage_name, nrow, nsamp)

if __name__ == '__main__':
    from device import Parser
    parser = Parser(description=__doc__)
    parser.add_argument('-r', '--nrow',default=650,type=int,
                        help='number of rows to read')
    parser.add_argument('-s', '--nsamp',default=1,type=int,
                        help='number of samples per exposure')
                        
    parser.add_argument('-v_file', '--voltage_file',default=None, type=string,
                        help='voltage file or path to voltage file')

    parser.add_argument('-v_name', '--voltage_name',default='v', type=string,
                        help='voltage name i.e. v, t, h, s, o, r, d')
    
    parser.add_argument('-all', '--run_all',default=False, type=bool,
                        help='Run all voltage scans at the same time')
            
    parser.add_verbose()
    parser.add_version()
    args = parser.parse_args()

    run_scan(voltage_file=args.voltage_file, voltage_name=args.voltage_name, nrow=args.nrow, nsamp=args.nsamp, run_all=args.run_all)
