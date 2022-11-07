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
    print('(',alpha,',',beta,',',gamma,')')
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

    dev_loc2nd = np.argmax(dev_locs > 0) + N

    min_err_loc = np.argmax(min_err > 0) + N
    max_err_loc = np.argmax(max_err > 0) + N

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



files = [#"/Volumes/GoogleDrive/Shared drives/Quantum Materials/MagnetTime/2022_02_TLH/VT69/day3/0.4K_60deg_sweep1.csv",
#"/Volumes/GoogleDrive/Shared drives/Quantum Materials/MagnetTime/2022_02_TLH/VT69/day3/0.4K_64deg_sweep1.csv",
"/Volumes/GoogleDrive/Shared drives/Quantum Materials/MagnetTime/2022_02_TLH/VT69/day3/0.4K_67.5deg_sweep1.csv",
"/Volumes/GoogleDrive/Shared drives/Quantum Materials/MagnetTime/2022_02_TLH/VT69/day3/0.4K_72deg_sweep1.csv",
"/Volumes/GoogleDrive/Shared drives/Quantum Materials/MagnetTime/2022_02_TLH/VT69/day3/0.4K_79deg_sweep1.csv",
"/Volumes/GoogleDrive/Shared drives/Quantum Materials/MagnetTime/2022_02_TLH/VT69/day3/0.4K_82.5deg_sweep1.csv",
"/Volumes/GoogleDrive/Shared drives/Quantum Materials/MagnetTime/2022_02_TLH/VT69/day3/0.3K_90deg_sweep1.csv"]

cmap = select_discrete_cmap('venasaur')

angles = ['67.5$^\mathregular{o}$', '72$^\mathregular{o}$', '79$^\mathregular{o}$', '82$^\mathregular{o}$','89$^\mathregular{o}$'
]#'60$^\mathregular{o}$', '64$^\mathregular{o}$',

angles_number = [67.5, 72, 79, 82, 89]#60, 64,

fig2, ax_scat = MakePlot(figsize=(6,6),ncols=1).create()

fig, a = MakePlot(figsize=(16,8), gs=True).create()

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
    B_inds1 = B > 1
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
    B_dev_anal, min_err_loc, max_err_loc = get_bsqr_deviation_analytic(field, volts, 500,plot=False, threshold=threshold)

    # ax1.scatter(angles_number[i], B[dev_loc])

    ax3.plot(B, tau - tau[0], linewidth=2, c=plt.cm.autumn(i/len(files)), label=str(angles[i]), alpha=.6)

    ax4.scatter(angles_number[i], B_dev_num, s=200, c=plt.cm.autumn(i/len(files)), alpha=.4)
    ax5.scatter(angles_number[i], B_dev_anal, s=200, c=plt.cm.autumn(i/len(files)), alpha=.4)
    ax_scat.scatter(angles_number[i], B_dev_anal, s=200, c='indianred', alpha=.9)

    all_angles.append(angles_number[i])
    all_bdevs.append(B_dev_anal)

    all_min_error.append(min_err_loc)
    all_max_error.append(max_err_loc)


main_path = r'/Volumes/GoogleDrive/Shared drives/Quantum Materials/MagnetTime/2022_04_CBR/'

files = ['PPMS_VT16_VT69_file00002.txt',
         'PPMS_VT16_VT69_file00005.txt',
         'PPMS_VT16_VT69_file00006.txt',
         'PPMS_VT16_VT69_file000010.txt',
         'PPMS_VT16_VT69_file000011.txt']

# Fix the 75 with the stupid shit file 3

T_75 = np.genfromtxt(main_path+'PPMS_VT16_VT69_file00003.txt', delimiter='\t', skip_header=(3))[:,0]

T_inds = T_75 < 14100.3870000000

B_75 = np.flip(np.genfromtxt(main_path+'PPMS_VT16_VT69_file00003.txt', delimiter='\t', skip_header=(3))[T_inds,9])
LCR_75 = np.flip(savgol_filter(median_filter(np.genfromtxt(main_path+'PPMS_VT16_VT69_file00003.txt', delimiter='\t', skip_header=(3))[T_inds,15],55),11,3))
AH_75 = np.flip(savgol_filter(median_filter(np.genfromtxt(main_path+'PPMS_VT16_VT69_file00003.txt', delimiter='\t', skip_header=(3))[T_inds,30],35),11,3))




# B_75 = np.genfromtxt(main_path+'PPMS_VT16_VT69_file00002.txt', delimiter='\t', skip_header=(3))[:,9]
# LCR_75 = savgol_filter(median_filter(np.genfromtxt(main_path+'PPMS_VT16_VT69_file00002.txt', delimiter='\t', skip_header=(3))[:,15],35),11,3)
# AH_75 = savgol_filter(median_filter(np.genfromtxt(main_path+'PPMS_VT16_VT69_file00002.txt', delimiter='\t', skip_header=(3))[:,30],35),11,3)

# AH and LCR swapped here, VT16 on AH from this point onwards

B_69 = np.genfromtxt(main_path+'PPMS_VT16_VT69_file00005.txt', delimiter='\t', skip_header=(3))[:,9]
LCR_69 = savgol_filter(median_filter(np.genfromtxt(main_path+'PPMS_VT16_VT69_file00005.txt', delimiter='\t', skip_header=(3))[:,15],55),11,3)
AH_69 = savgol_filter(median_filter(np.genfromtxt(main_path+'PPMS_VT16_VT69_file00005.txt', delimiter='\t', skip_header=(3))[:,30],35),11,3)

B_60 = np.genfromtxt(main_path+'PPMS_VT16_VT69_file00010.txt', delimiter='\t', skip_header=(3))[:,9]
LCR_60 = savgol_filter(median_filter(np.genfromtxt(main_path+'PPMS_VT16_VT69_file00010.txt', delimiter='\t', skip_header=(3))[:,15],55),11,3)
AH_60 = savgol_filter(median_filter(np.genfromtxt(main_path+'PPMS_VT16_VT69_file00010.txt', delimiter='\t', skip_header=(3))[:,30],35),11,3)

B_53a = np.genfromtxt(main_path+'PPMS_VT16_VT69_file00015.txt', delimiter='\t', skip_header=(3))[:,9]
LCR_53a = savgol_filter(median_filter(np.genfromtxt(main_path+'PPMS_VT16_VT69_file00015.txt', delimiter='\t', skip_header=(3))[:,15],55),11,3)
AH_53a = savgol_filter(median_filter(np.genfromtxt(main_path+'PPMS_VT16_VT69_file00015.txt', delimiter='\t', skip_header=(3))[:,30],35),11,3)

B_53b = np.genfromtxt(main_path+'PPMS_VT16_VT69_file00018.txt', delimiter='\t', skip_header=(3))[:,9]
LCR_53b = savgol_filter(median_filter(np.genfromtxt(main_path+'PPMS_VT16_VT69_file00018.txt', delimiter='\t', skip_header=(3))[:,15],101),21,3)
AH_53b = savgol_filter(median_filter(np.genfromtxt(main_path+'PPMS_VT16_VT69_file00018.txt', delimiter='\t', skip_header=(3))[:,30],35),11,3)


B_55 = np.genfromtxt(main_path+'PPMS_VT16_VT69_file00047.txt', delimiter='\t', skip_header=(3))[:,9]
LCR_55 = savgol_filter(median_filter(np.genfromtxt(main_path+'PPMS_VT16_VT69_file00047.txt', delimiter='\t', skip_header=(3))[:,15],101),21,3)
AH_55 = savgol_filter(median_filter(np.genfromtxt(main_path+'PPMS_VT16_VT69_file00047.txt', delimiter='\t', skip_header=(3))[:,30],35),11,3)


T_47 = np.genfromtxt(main_path+'PPMS_VT16_VT69_file00024.txt', delimiter='\t', skip_header=(4))[:,0]

T_inds_47 = T_47 < 14003.8770000000

B_47 = np.genfromtxt(main_path+'PPMS_VT16_VT69_file00024.txt', delimiter='\t', skip_header=(4))[T_inds_47,9]
LCR_47 = savgol_filter(median_filter(np.genfromtxt(main_path+'PPMS_VT16_VT69_file00024.txt', delimiter='\t', skip_header=(4))[T_inds_47,15],253),45,3)
AH_47 = savgol_filter(median_filter(np.genfromtxt(main_path+'PPMS_VT16_VT69_file00024.txt', delimiter='\t', skip_header=(4))[T_inds_47,30],35),11,3)

B_43 = np.genfromtxt(main_path+'PPMS_VT16_VT69_file00027.txt', delimiter='\t', skip_header=(4))[:,9]
LCR_43 = savgol_filter(median_filter(np.genfromtxt(main_path+'PPMS_VT16_VT69_file00027.txt', delimiter='\t', skip_header=(4))[:,15],253),45,3)
AH_43 = savgol_filter(median_filter(np.genfromtxt(main_path+'PPMS_VT16_VT69_file00027.txt', delimiter='\t', skip_header=(4))[:,30],35),11,3)

B_45 = np.genfromtxt(main_path+'PPMS_VT16_VT69_file00032.txt', delimiter='\t', skip_header=(4))[:,9]
LCR_45 = savgol_filter(median_filter(np.genfromtxt(main_path+'PPMS_VT16_VT69_file00032.txt', delimiter='\t', skip_header=(4))[:,15],35),45,3)
AH_45 = savgol_filter(median_filter(np.genfromtxt(main_path+'PPMS_VT16_VT69_file00032.txt', delimiter='\t', skip_header=(4))[:,30],35),11,3)

B_39 = np.genfromtxt(main_path+'PPMS_VT16_VT69_file00040.txt', delimiter='\t', skip_header=(4))[:,9]
LCR_39 = savgol_filter(median_filter(np.genfromtxt(main_path+'PPMS_VT16_VT69_file00040.txt', delimiter='\t', skip_header=(4))[:,15],101),45,3)
AH_39 = savgol_filter(median_filter(np.genfromtxt(main_path+'PPMS_VT16_VT69_file00040.txt', delimiter='\t', skip_header=(4))[:,30],35),11,3)

B_29 = np.genfromtxt(main_path+'PPMS_VT16_VT69_file00054.txt', delimiter='\t', skip_header=(4))[:,9]
LCR_29 = savgol_filter(median_filter(np.genfromtxt(main_path+'PPMS_VT16_VT69_file00054.txt', delimiter='\t', skip_header=(4))[:,15],101),45,3)
AH_29 = savgol_filter(median_filter(np.genfromtxt(main_path+'PPMS_VT16_VT69_file00054.txt', delimiter='\t', skip_header=(4))[:,30],35),11,3)

B_13 = np.genfromtxt(main_path+'PPMS_VT16_VT69_file00059.txt', delimiter='\t', skip_header=(4))[:,9]
LCR_13 = savgol_filter(median_filter(np.genfromtxt(main_path+'PPMS_VT16_VT69_file00059.txt', delimiter='\t', skip_header=(4))[:,15],101),45,3)
AH_13 = savgol_filter(median_filter(np.genfromtxt(main_path+'PPMS_VT16_VT69_file00059.txt', delimiter='\t', skip_header=(4))[:,30],35),11,3)

Bs = [B_75, B_69, B_60, B_55, B_45, B_39, B_29]#, B_13]

VT16s = [AH_75, AH_69, AH_60, AH_55, AH_45, LCR_39, AH_29]#, AH_13]
VT69s = [LCR_75, LCR_69, LCR_60, AH_53b, AH_47, AH_43, LCR_39]
angles = [75, 69, 60, 55, 45, 39, 29]#, 13]



for i, B in enumerate(Bs):
    flip=False
    if B[0] > B[-1]:
        flip = True
        B = np.flip(B)

    x = B[100:]
    B = x
    if flip:
        VT16s[i] = np.flip(VT16s[i])
    y = VT16s[i][100:]-VT16s[i][100]

    tau = 1e3*median_filter(y, med_num)

    tau_copy = tau

    field = x
    volts = tau

    volts -= volts[0]

    if volts[0] > volts[10]:
        volts *= -1

    B_dev_num = get_bsqr_deviation_numerically(field, volts, 10, threshold=threshold_numeric)
    B_dev_anal, min_err_loc, max_err_loc = get_bsqr_deviation_analytic(field, volts, 10, plot=False, threshold=threshold)


    ax1.plot(B, tau - tau[0], linewidth=2, c=plt.cm.winter(i / len(files)), label=str(angles[i]) + '$^\mathregular{o}$', alpha=.6)

    ax4.scatter(angles[i], B_dev_num, s=200, c=plt.cm.winter(i / len(files)), alpha=.4)
    ax5.scatter(angles[i], B_dev_anal, s=200, c=plt.cm.winter(i / len(files)), alpha=.4)
    ax_scat.scatter(angles[i], B_dev_anal, s=200, c='indianred', alpha=.9)
    all_angles.append(angles[i])
    all_bdevs.append(B_dev_anal)
    all_min_error.append(min_err_loc)
    all_max_error.append(max_err_loc)



B_2 = np.genfromtxt(main_path+'D_graf/2deg.dat', delimiter='\t', skip_header=(4), dtype=np.float64)[:,0]
tau_2 = np.genfromtxt(main_path+'D_graf/2deg.dat', delimiter='\t', skip_header=(4), dtype=np.float64)[:,1]
# tau_2 = savgol_filter(median_filter(np.genfromtxt(main_path+'D_graf/2deg.dat', delimiter='\t', skip_header=(4))[:,1],35),11,3)

B_9 = np.genfromtxt(main_path+'D_graf/9deg.dat', delimiter='\t', skip_header=(4), dtype=np.float64)[:,0]
tau_9 = np.genfromtxt(main_path+'D_graf/9deg.dat', delimiter='\t', skip_header=(4), dtype=np.float64)[:,1]
# tau_9 = savgol_filter(median_filter(np.genfromtxt(main_path+'D_graf/9deg.dat', delimiter='\t', skip_header=(4))[:,1],35),11,3)

B_12p5 = np.genfromtxt(main_path+'D_graf/12p5deg.dat', delimiter='\t', skip_header=(4), dtype=np.float64)[:,0]
tau_12p5 = np.genfromtxt(main_path+'D_graf/12p5deg.dat', delimiter='\t', skip_header=(4), dtype=np.float64)[:,1]
# tau_12p5 = savgol_filter(median_filter(np.genfromtxt(main_path+'D_graf/12p5deg.dat', delimiter='\t', skip_header=(4))[:,1],35),11,3)

B_16 = np.genfromtxt(main_path+'D_graf/16deg.dat', delimiter='\t', skip_header=(4), dtype=np.float64)[:,0]
tau_16 = np.genfromtxt(main_path+'D_graf/16deg.dat', delimiter='\t', skip_header=(4), dtype=np.float64)[:,1]
# tau_16 = savgol_filter(median_filter(np.genfromtxt(main_path+'D_graf/16deg.dat', delimiter='\t', skip_header=(4))[:,1],35),11,3)

Bs = [B_2, B_9, B_12p5, B_16]

VT16s = [tau_2, tau_9, tau_12p5, tau_16]

angles_tlh = [2,9,12.5,16]

vt16_dict = {}

B_uppers = [36,28,23,22]

for i, B in enumerate(Bs):

    flip = False
    if B[0] > B[-1]:
        flip = True
        B = np.flip(B)

    B_copy = B
    B_inds1 = B > 8
    B_inds2 = B < B_uppers[i]
    B_inds = B_inds1 & B_inds2
    B = B[B_inds]
    x = B
    B = x
    if flip:
        VT16s[i] = np.flip(VT16s[i])
    y = VT16s[i][B_inds]
    y -= y[0]



    tau = median_filter(y, med_num)

    tau_copy = tau

    tau = 1e4*tau

    field = x
    volts = tau

    volts -= volts[0]

    if volts[0] > volts[10]:
        volts *= -1

    B_dev_num = get_bsqr_deviation_numerically(field, volts, 500, threshold=threshold_numeric)
    B_dev_anal, min_err_loc, max_err_loc = get_bsqr_deviation_analytic(field, volts, 500, plot=False, threshold=threshold)

    ax2.plot(B, tau - tau[0], linewidth=2, c=plt.cm.spring(i / len(files)), label=str(angles_tlh[i]) + '$^\mathregular{o}$', alpha=.6)

    ax4.scatter(angles_tlh[i], B_dev_num, s=200, c=plt.cm.spring(i / len(files)), alpha=.4)
    ax5.scatter(angles_tlh[i], B_dev_anal, s=200, c=plt.cm.spring(i / len(files)), alpha=.4)
    ax_scat.scatter(angles_tlh[i], B_dev_anal, s=200, c='indianred', alpha=.9)

    all_angles.append(angles_tlh[i])
    all_bdevs.append(B_dev_anal)
    all_min_error.append(min_err_loc)
    all_max_error.append(max_err_loc)


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

print(all_angles_array)
print(all_max)
print(all_min)
yerr = all_max/2
# yerr /=2

# yerr = all_min
fit = np.poly1d(np.polyfit(all_angles_array, all_bdevs_array, 4))

linspace = np.linspace(0,98)

ax_scat.plot(linspace,fit(linspace), linewidth=2,c='darkslategray')
ax_scat.errorbar(all_angles_array, all_bdevs_array, yerr=yerr, fmt='none', c='k', linewidth=2.1)
plt.tight_layout(pad=1.5)


df = pd.DataFrame(np.array([all_angles_array, all_bdevs_array, yerr]).T,columns=['Angle', 'B_dev', 'yerr'])

df.to_csv(main_path+'B2_dev_c-a.csv', index=False)

publication_plot(ax1, 'Magnetic Field (T)', r'Capacitance (arb.)', title='VT16 PPMS')
publication_plot(ax2, 'Magnetic Field (T)', '', title='VT16 TLH June')
publication_plot(ax3, 'Magnetic Field (T)', '', title='VT69 TLH March')
publication_plot(ax4, r'$\theta$', '$B_{\mathrm{Numerical}}$')
publication_plot(ax5, r'$\theta$', '$B_{\mathrm{Analytical}}$')
publication_plot(ax_scat, r'$\theta$', '$B_{\mathrm{Analytical}}$')
plt.tight_layout(pad=1.5)


legend = ax1.legend(framealpha=0, ncol=2, loc='upper left',
              prop={'size': 24, 'family': 'arial'}, handlelength=0, columnspacing = .5)

for line,text in zip(legend.get_lines(), legend.get_texts()):
    text.set_color(line.get_color())

handles, labels = ax2.get_legend_handles_labels()


legend = ax2.legend(handles[::-1], labels[::-1], framealpha=0, ncol=1, loc='best',
              prop={'size': 24, 'family': 'arial'}, handlelength=0)

for line,text in zip(np.flip(legend.get_lines()), np.flip(legend.get_texts())):
    text.set_color(line.get_color())

legend = ax3.legend(framealpha=0, ncol=2, loc='best',
              prop={'size': 24, 'family': 'arial'}, handlelength=0, columnspacing=0.5)

for line,text in zip(legend.get_lines(), legend.get_texts()):
    text.set_color(line.get_color())

bbox_args = dict(boxstyle="round", fc="0.8")
arrow_args = dict(width=0.8, headlength=10, headwidth=8,color='k')

trans = ax4.get_xaxis_transform() # x in data untis, y in axes fraction
# ann = ax.annotate('MgII', xy=(2000, 1.05 ), xycoords=trans)

ax4.annotate(r'$[101]$', xy=(62.3, -0.01), xycoords=trans,
                 xytext=(62.3, -0.3),
                 ha="left", va="center",
                 arrowprops=arrow_args, fontname='arial', fontsize=22)

ax4.annotate(r'$(\overline{1}01)_\perp$', xy=(27.7, -0.01), xycoords=trans,
                 xytext=(27.7, -0.3),
                 ha="right", va="center",
                 arrowprops=arrow_args, fontname='arial', fontsize=22)

trans = ax5.get_xaxis_transform() # x in data untis, y in axes fraction

ax5.annotate(r'$[101]$', xy=(62.3, -0.01), xycoords=trans,
                 xytext=(62.3, -0.3),
                 ha="left", va="center",
                 arrowprops=arrow_args, fontname='arial', fontsize=22)
ax5.annotate(r'$(\overline{1}01)_\perp$', xy=(27.7, -0.01), xycoords=trans,
                 xytext=(27.7, -0.3),
                 ha="right", va="center",
                 arrowprops=arrow_args, fontname='arial', fontsize=22)

plt.tight_layout(pad=.8)
plt.show()



