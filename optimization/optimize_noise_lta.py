#!/usr/bin/env python
import os, sys, time, datetime
import glob
import logging
import numpy as np

sys.path.append('path/to/device')
from lta import Lta


def take_image(exptime=0.0,nrow=None,nsamp=None):
    """ Peform readout 

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

def get_dir_name():
    today = datetime.now()
    if today.hour < 12:
        h = "00"
    else:
        h = "12"
    
    return ("./images/xxx/" + today.strftime('%Y%m%d')+ h +"_"+"noise optimization")

def noise_scan(psamp_ssamp,nrow,nsamp):

    """ Peforms noise scan with given psamp, nsamp values 
    
    Parameters:
    -----------
    psamp_ssamp : values of psamp and ssamp (psamp=ssamp) to peform the scan
    nrow : number of rows to read ('lta NROW')
    nsamp : number of samples to read ('lta NSAMP')

    Returns:
    --------
    None
    """
    
    os.mkdir(get_dir_name())
    for value in psamp_ssamp:
        Lta.talk("name",get_dir_name() + "/psamp_ssamp_%d"%value)
        Lta.set('psamp',value)
        Lta.set('ssamp', value)
        take_image(nrow=nrow, nsamp=nsamp)

if __name__ == '__main__':
    from device import Parser
    parser = Parser(description=__doc__)
    parser.add_argument('-r', '--nrow',default=650,type=int,
                        help='number of rows to read')
    parser.add_argument('-s', '--nsamp',default=1,type=int,
                        help='number of samples per exposure')
                        
    parser.add_argument('-p_s', '--psamp_ssamp',default=None, type=list,
                        help='psamp_ssamp values in a list i.e. [a,b,c, ...]')
    parser.add_verbose()
    parser.add_version()
    args = parser.parse_args()

    noise_scan(psamp_ssamp=args.psamp_ssamp,nsamp=args.nsamp,row=args.nrow)
       




