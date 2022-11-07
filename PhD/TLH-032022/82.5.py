import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter, argrelmin, argrelmax
from scipy.ndimage import median_filter
from scipy.optimize import curve_fit
from tools.utils import *
from tools.ColorMaps import select_discrete_cmap



def get_bsqr_deviation_numerically(field, volts, N, threshold=3.,savgol_num=501,savgol_poly=2, return_third_deriv=False,plot=False):

    # first numerically get the value where its much bigger than 2nd deriv avg

    first_deriv = savgol_filter(volts, savgol_num, savgol_poly,deriv=1)
    second_deriv = savgol_filter(first_deriv,savgol_num,savgol_poly,deriv=1)
    third_deriv = savgol_filter(second_deriv, savgol_num, 2, deriv=1)

    mean_2nd_deriv_og = np.mean(second_deriv[N:2*N])
    mean_3rd_deriv_og = np.mean(third_deriv[N:2 * N])

    dev_locs = np.abs(second_deriv) > threshold*np.abs(mean_2nd_deriv_og) #+ 25 * std_2nd_deriv_og
    dev_locs3 = np.abs(third_deriv) > threshold*np.abs(mean_3rd_deriv_og) #+ 25 * std_2nd_deriv_og


    dev_loc2nd = np.argmax(dev_locs > 0)
    dev_loc3rd = np.argmax(dev_locs3 > 0)

    if plot:
        fig, axs = plt.subplots(ncols=2,figsize=(16,9))
        axs[0].plot(field, volts,linewidth=2,c='indianred',label='Data')
        # axs[0].plot(field, func(field,alpha, beta, gamma),linewidth=2,c='midnightblue',label='Fit',linestyle='dashed')

        axs[1].plot(field, second_deriv,linewidth=2,c='darkgray')
        axs[1].axvline(field[dev_loc2nd])
        axs[0].axvline(field[dev_loc2nd])

        axs[0].legend(framealpha=0, ncol=1, loc='best',
                            prop={'size': 24, 'family': 'arial'})
        publication_plot(axs[0],'Magnetic Field (T)', 'Torque (arb.)')
        publication_plot(axs[1], 'Magnetic Field (T)', r'$\frac{\partial^2 \tau}{\partial B^2}$')
        plt.tight_layout(pad=1)
        plt.show()


    if return_third_deriv:
        return field[dev_loc2nd], field[dev_loc3rd]
    else:
        return field[dev_loc2nd]


def get_bsqr_deviation_analytic(field, volts, N, plot=False,threshold=3.,std=False):

    func = lambda x, alpha, beta, gamma : alpha * x ** 2 + beta * np.exp(x * gamma)

    # func = lambda x, alpha, beta, gamma : alpha * x ** 2 + beta / (x - gamma)

    popt, pcov = curve_fit(func, field, volts)

    alpha = popt[0]
    beta = popt[1]
    gamma = popt[2]
    err_in_fit = np.sqrt(np.diag(pcov))

    err_in_deriv = 2*err_in_fit[0] + np.sqrt((err_in_fit[1]/beta)**2 + 2*(err_in_fit[2]/gamma)**2)

    second_deriv = 2 * alpha + beta * gamma ** 2 * np.exp(gamma * field)

    mean_2nd_deriv_og = np.mean(second_deriv[:N])
    std_2nd_deriv_og = np.std(second_deriv[:N])

    locs_of_mean = np.arange(N)
    locs = np.setdiff1d(np.arange(len(field)), locs_of_mean)



    if not std:
        dev_locs = np.abs(second_deriv)[locs] > threshold * np.abs(mean_2nd_deriv_og)  # + 25 * std_2nd_deriv_og
    else:
        dev_locs = np.abs(second_deriv)[locs] > threshold * std_2nd_deriv_og

    min_err =  np.abs(second_deriv)[locs] > threshold * np.abs(mean_2nd_deriv_og) - err_in_deriv
    max_err =  np.abs(second_deriv)[locs] > threshold * np.abs(mean_2nd_deriv_og) + err_in_deriv

    dev_loc2nd = np.argmax(dev_locs > 0)

    min_err_loc = np.argmax(min_err > 0)
    max_err_loc = np.argmax(max_err > 0)

    if plot:
        fig, axs = plt.subplots(ncols=2,figsize=(16,9))
        axs[0].plot(field, volts,linewidth=2,c='indianred',label='Data')
        axs[0].plot(field, func(field,alpha, beta, gamma),linewidth=2,c='midnightblue',label='Fit',linestyle='dashed')

        axs[1].plot(field, second_deriv,linewidth=2,c='darkgray')
        axs[1].axvline(field[dev_loc2nd])
        axs[0].axvline(field[dev_loc2nd])

        axs[0].legend(framealpha=0, ncol=1, loc='best',
                            prop={'size': 24, 'family': 'arial'})
        publication_plot(axs[0],'Magnetic Field (T)', 'Torque (arb.)')
        publication_plot(axs[1], 'Magnetic Field (T)', r'$\frac{\partial^2 \tau}{\partial B^2}$')
        plt.tight_layout(pad=1)
        plt.show()



    return field[dev_loc2nd], field[min_err_loc], field[max_err_loc]



files = [#"G:/Shared drives/Quantum Materials/MagnetTime/2022_02_TLH/VT69/day3/0.4K_60deg_sweep1.csv",
# "G:/Shared drives/Quantum Materials/MagnetTime/2022_02_TLH/VT69/day3/0.4K_64deg_sweep1.csv",
# "G:/Shared drives/Quantum Materials/MagnetTime/2022_02_TLH/VT69/day3/0.4K_67.5deg_sweep1.csv",
# "G:/Shared drives/Quantum Materials/MagnetTime/2022_02_TLH/VT69/day3/0.4K_72deg_sweep1.csv",
# "G:/Shared drives/Quantum Materials/MagnetTime/2022_02_TLH/VT69/day3/0.4K_79deg_sweep1.csv",
'/Users/npopiel/Desktop/82/0.4K_82.5deg_sweep1.csv']
# "G:/Shared drives/Quantum Materials/MagnetTime/2022_02_TLH/VT69/day3/0.3K_90deg_sweep1.csv"]

cmap = select_discrete_cmap('venasaur')

angles = ['60$^\mathregular{o}$', '64$^\mathregular{o}$', '67.5$^\mathregular{o}$', '72$^\mathregular{o}$', '79$^\mathregular{o}$', '82$^\mathregular{o}$','89$^\mathregular{o}$'
]

angles_number = [60, 64, 67.5, 72, 79, 82, 89]

fig2, ax_scat = plt.subplots(figsize=(12,6),ncols=2)

fig = plt.figure(figsize=(16,8))

all_angles, all_bdevs = [], []
all_min_error, all_max_error = [], []

gs = fig.add_gridspec(2,4)
ax1 = fig.add_subplot(gs[:, 0])
ax2 = fig.add_subplot(gs[:, 1])
ax3 = fig.add_subplot(gs[:, 2])
ax4 = fig.add_subplot(gs[0, 3])
ax5 = fig.add_subplot(gs[1, 3])


med_num, savgol_num = 51, 501
N=50
threshold=1.001
threshold_numeric = 6
for i, file in enumerate(files):

    B = np.genfromtxt(file, delimiter=',', skip_header=4)[:,0]
    flip=False
    if B[0] > B[-1]:
        flip = True
        B = np.flip(B)
    B_copy = B
    B_inds1 = B > 8
    B_inds2 = B < 32
    B_inds = B_inds1 & B_inds2
    B = B[B_inds]
    x=B

    # tau = savgol_filter(median_filter(1e3*np.genfromtxt(file, delimiter='\t', skip_header=4)[:,1],med_num),savgol_num,5)

    tau = median_filter(1e3 * np.genfromtxt(file, delimiter=',', skip_header=4)[:, 1], med_num)
    if flip:
        tau = np.flip(tau)

    tau_copy = tau

    tau = tau[B_inds]

    field = B
    volts = tau

    volts -= volts[0]

    if volts[0] > volts[10]:

        volts *= -1


    B_dev_num = get_bsqr_deviation_numerically(field, volts, 500, threshold=threshold_numeric, plot=False)
    B_dev_anal, min_err_loc, max_err_loc = get_bsqr_deviation_analytic(field, volts, 500,plot=True, threshold=threshold)

    # ax1.scatter(angles_number[i], B[dev_loc])

    ax3.plot(B, tau - tau[0], linewidth=2, c=plt.cm.autumn(i/len(files)), label=str(angles[i]), alpha=.6)

    ax4.scatter(angles_number[i], B_dev_num, s=200, c=plt.cm.autumn(i/len(files)), alpha=.4)
    ax5.scatter(angles_number[i], B_dev_anal, s=200, c=plt.cm.autumn(i/len(files)), alpha=.4)
    ax_scat[0].scatter(angles_number[i], B_dev_num, s=200, c='indianred', alpha=.9)
    ax_scat[1].scatter(angles_number[i], B_dev_anal, s=200, c='indianred', alpha=.9)

    all_angles.append(angles_number[i])
    all_bdevs.append(B_dev_anal)

    all_min_error.append(min_err_loc)
    all_max_error.append(max_err_loc)


main_path = r'G:/Shared drives/Quantum Materials/MagnetTime/2022_04_CBR/'

files = ['PPMS_VT16_VT69_file00002.txt',
         'PPMS_VT16_VT69_file00005.txt',
         'PPMS_VT16_VT69_file00006.txt',
         'PPMS_VT16_VT69_file000010.txt',
         'PPMS_VT16_VT69_file000011.txt']


ax1.set_xticks([0,5,10,15])

# ax.set_xticks([0,5,10,15])
# ax.set_yticks([4,5,6,7,8])
# ax4.set_yticks([6,8,10,12,14])
ax4.set_xticks([0, 20, 40, 60, 80,100])
ax4.set_xticklabels([0, 20, 40, '', 80,100])

ax5.set_xticks([0, 20, 40, 60, 80, 100])
ax5.set_xticklabels([0, 20, 40, '', 80, 100])


sorted_inds = np.argsort(all_angles)

all_angles_array = np.array(all_angles)[sorted_inds]
all_bdevs_array = np.array(all_bdevs)[sorted_inds]

all_min = np.array(all_min_error)[sorted_inds]
all_max = np.array(all_max_error)[sorted_inds]

# yerr = np.array([all_min, all_max])



plt.tight_layout(pad=.8)
plt.show()



