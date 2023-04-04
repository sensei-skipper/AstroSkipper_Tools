import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, peak_widths


def get_peaks(sclipData, plot=True):
    param = list()

    fig = plt.figure(figsize=(5,2), dpi= 175, facecolor='w', edgecolor='k')
    
    n, bins, patches = plt.hist(sclipData,bins=np.linspace(-200,600,160), density=False, fc='orange',histtype='step')
   
    # determine the indices of the local maxima
    peaks, _ = find_peaks(n, height=20, distance=10, width=0.6, prominence=95) 
    peakval = bins[peaks]
    peakdiff = np.diff(peakval)
    gain = np.mean(peakdiff)
    w = peak_widths(n, peaks, rel_height=0.01)[0]
    peakLoc = peakval/gain
    

    plt.grid()
    plt.xlabel('Pixel Value (ADUs)')
    plt.ylabel('Entries')
    plt.plot(peakval, n[peaks], "x",color='red',lw=200)
    plt.title(r"$K \approx {:.2f} ADUs/e^-$".format(gain),fontsize=15)
    
    if plot:
        plt.show()
        
    else:
        plt.close()
    
    for i in range(len(peaks)):
        param.append(bins[peaks[i]])       #mean
        param.append(w[i])                 #std
        param.append(n[peaks[i]])          #amplitude
    
    return np.array(param), len(peaks),gain
