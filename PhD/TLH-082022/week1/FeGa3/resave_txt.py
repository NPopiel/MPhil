import numpy as np

filenames = ['0.29K_46.66deg_sweep056_down.csv',
'0.29K_-6deg_sweep066_down.csv',
'0.29K_-6deg_sweep069_up.csv',
'0.29K_-6deg_sweep070_up.csv',
'0.29K_-6deg_sweep074_up.csv',
'0.29K_-12deg_sweep078_down.csv',
'0.29K_26deg_sweep040_down.csv',
'0.29K_26deg_sweep041_down.csv',
'0.29K_33.33deg_sweep046_down.csv',
'0.29K_33.33deg_sweep048_up.csv',
'0.29K_40deg_sweep051_down.csv',
'0.29K_40deg_sweep052_up.csv',
'0.29K_46.66deg_sweep055_down.csv']

path = '/Volumes/GoogleDrive/Shared drives/Quantum Materials/MagnetTime/2022_08_TLH/week1/FeGa3-VLS1/good/15up/'

for filename in filenames:

    dat = np.genfromtxt(path+filename, delimiter=',')

    np.savetxt(path+'mega/'+filename.split('.')[0]+'.txt', dat, delimiter='\t')