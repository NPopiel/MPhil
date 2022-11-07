import numpy as np

filenames = ['0.3K_104deg_sweep122_up.csv']

path = '/Volumes/GoogleDrive/Shared drives/Quantum Materials - Sebastian Group/MagnetTime/2022_10_TLH/week2/torque/VT15/day3/'

save_path = '/Volumes/GoogleDrive/Shared drives/Quantum Materials - Sebastian Group/MagnetTime/2022_10_TLH/week2/torque/VT15-mega/'


for filename in filenames:

    dat = np.genfromtxt(path+filename, delimiter=',')


    np.savetxt(save_path+filename.split('csv')[0]+'txt', dat, delimiter='\t')