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


def get_bsqr_deviation_analytic(field, volts, N, plot=False,threshold=8.,std=True):

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
        dev_locs = np.abs(second_deriv)[locs] > threshold * alpha

    min_err =  np.abs(second_deriv)[locs] > threshold * np.abs(mean_2nd_deriv_og) - err_in_deriv
    max_err =  np.abs(second_deriv)[locs] > threshold * np.abs(mean_2nd_deriv_og) + err_in_deriv

    dev_loc2nd = np.argmax(dev_locs > 0) + N

    min_err_loc = np.argmax(min_err > 0) + N
    max_err_loc = np.argmax(max_err > 0) + N

    if plot:
        fig, axs = MakePlot(ncols=2,figsize=(16,9)).create()
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


def get_bsqr_deviation_analytic2(field, volts, N, plot=False,threshold=8.,std=False):

    func = lambda x, alpha, beta, gamma, delta : alpha * x ** 2 + beta * np.exp(x * gamma) + delta

    # func = lambda x, alpha, beta, gamma : alpha * x ** 2 + beta / (x - gamma)

    popt, pcov = curve_fit(func, field, volts)

    alpha = popt[0]
    beta = popt[1]
    gamma = popt[2]
    delta = popt[3]
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
        dev_locs = np.abs(second_deriv)[locs] > threshold * alpha

    min_err =  np.abs(second_deriv)[locs] > threshold * np.abs(mean_2nd_deriv_og) - err_in_deriv
    max_err =  np.abs(second_deriv)[locs] > threshold * np.abs(mean_2nd_deriv_og) + err_in_deriv

    dev_loc2nd = np.argmax(dev_locs > 0) + N

    min_err_loc = np.argmax(min_err > 0) + N
    max_err_loc = np.argmax(max_err > 0) + N

    if plot:
        fig, axs = MakePlot(ncols=2,figsize=(16,9)).create()
        axs[0].plot(field, volts,linewidth=2,c='indianred',label='Data')
        axs[0].plot(field, func(field,alpha, beta, gamma, delta),linewidth=2,c='midnightblue',label='Fit',linestyle='dashed')

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




all_angles, all_bdevs = [], []
all_min_error, all_max_error = [], []

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
#VT69s = [LCR_75, LCR_69, LCR_60, AH_53b, AH_47, AH_43, LCR_39]
VT69s = [LCR_75, LCR_69, LCR_60, AH_55, AH_45, AH_39, LCR_29]
angles = [75, 69, 60, 55, 45, 39, 29]#, 13]

med_num, savgol_num = 51, 501
N=50
threshold=3
threshold_numeric = 3


for i, B in enumerate(Bs):
    flip=False
    if B[0] > B[-1]:
        flip = True
        B = np.flip(B)

    x = B[100:]
    B = x
    if flip:
        VT69s[i] = np.flip(VT69s[i])
    y = VT69s[i][100:]-VT69s[i][100]

    tau = 1e3*median_filter(y, med_num)

    tau_copy = tau

    field = x
    volts = tau

    volts -= volts[0]

    if volts[0] > volts[10]:
        volts *= -1

    B_dev_num = get_bsqr_deviation_numerically(field, volts, 10, threshold=threshold_numeric)
    B_dev_anal, min_err_loc, max_err_loc = get_bsqr_deviation_analytic(field, volts, 10, plot=True, std=True,threshold=threshold)


    all_angles.append(angles[i])
    all_bdevs.append(B_dev_anal)
    all_min_error.append(min_err_loc)
    all_max_error.append(max_err_loc)



sorted_inds = np.argsort(all_angles)

all_angles_array = np.array(all_angles)[sorted_inds]
all_bdevs_array = np.array(all_bdevs)[sorted_inds]

all_min = np.array(all_min_error)[sorted_inds]
all_max = np.array(all_max_error)[sorted_inds]


df = pd.DataFrame(np.array([all_angles_array, all_bdevs_array, all_max/2]).T, columns=['Angle', 'B_dev', 'yerr'])

df.to_csv(r'/Volumes/GoogleDrive/Shared drives/Quantum Materials/MagnetTime/2022_04_CBR/B2_dev_c-aVT69PPMS.csv', index=False)



