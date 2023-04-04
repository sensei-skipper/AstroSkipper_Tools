import sys 
from pylab import *
import pylab as plt
from scipy.optimize import curve_fit
import numpy as np
sys.path.append('/data/des81.b/data/emarrufo/AstroSkipper_Analysis/full_CCD_characterization/')
import peak_finding_algorithm_electrons

def gaussian(x,mu,sigma,A):
    """Gaussian"""
    return A*exp(-(x-mu)**2/2/sigma**2)

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
    params,cov=curve_fit(SuperPosition,x,y,p0=guess.flatten(),maxfev = 10000000)
    sigma=sqrt(diag(cov))
    return params, cov
def fit_multi_gaussian(clipped_data,min_b,max_b,plot=False):
    param, lenpeaks, k =peak_finding_algorithm_electrons.get_peaks(clipped_data,min_b,max_b, plot=False)
    bins=np.linspace(min_b,max_b,150)
    vals,edges = np.histogram(clipped_data,bins,density=False)
    centers = (edges[1:]+edges[:-1])/2. 
    guess = np.array(param).reshape(lenpeaks,3)
    # Perform the fit
    
    params, cov = MultiGaussian(centers,vals,guess)
    
    # Reshape the output parameters to match the guess
    
    params = np.array(params).reshape(guess.shape)

    if plot:
        fig, ax = plt.subplots(figsize=(8, 8))
        plt.hist(clipped_data,density=False,bins=bins,histtype='step')
        plt.plot(np.linspace(min_b,max_b,150), SuperPosition(np.linspace(min_b,max_b,150),*params.flatten()),color='green',lw=4,alpha=0.5)
        ax.set_xlabel('Charge ($\mathregular{e^{-}}$)', fontsize=30)
        ax.set_ylabel('Entries',fontsize=30)
        ax.tick_params(axis='both', which='major', labelsize=20)
        plt.grid()
        plt.show()




