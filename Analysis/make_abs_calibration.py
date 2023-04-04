import numpy as np 
wave_photodiode, responsivity = np.loadtxt('./abs_calibration/thorlabs_photo_diode_calibration_full.txt',
                                            usecols=(0,1), 

                                            skiprows=(1), 
                                           unpack=True,delimiter=",")
def get_wave():
    return wave_photodiode

def calculate_power(photo_current,responsivity):
    return photo_current/responsivity

def get_cube_power():
    cube_full_data = np.abs(np.load('./abs_calibration/ABS_cal_full_cube.npy'))
    cube_full_data_2 = np.abs(np.load('./abs_calibration/ABS_cal_full_cube2.npy'))
    cube_combined = np.stack((cube_full_data,cube_full_data_2), axis=1).flatten()
    cube_combined = cube_combined.reshape(len(wave_photodiode), int(len(cube_combined)/len(wave_photodiode)))
    cube_combined =  np.median(cube_combined,axis=1)
    cube_power_combined = calculate_power(cube_combined,responsivity)
    area_ratio = (0.01)**2 / (1.5e-5)**2 
    
    return cube_power_combined /(area_ratio)

def get_sphere_power():
    sphere_power = np.abs(np.load('./abs_calibration/ABS_cal_full_sphere.npy'))
    sphere_power_2 = np.abs(np.load('./abs_calibration/ABS_cal_full_sphere2.npy'))
    sphere_combined=np.stack((sphere_power,sphere_power_2), axis=1).flatten()
    sphere_combined= sphere_combined.reshape(len(wave_photodiode), int(len(sphere_combined)/len(wave_photodiode)))
    return   np.median(sphere_combined,axis=1)



def get_calibration():
    
    calibration = get_cube_power()/get_sphere_power()
    data = np.column_stack([get_wave(), calibration])
    np.savetxt=("Absolute_Calibration.txt",data)
    return calibration
    
   

