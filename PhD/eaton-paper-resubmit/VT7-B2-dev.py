import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter, argrelmin, argrelmax
from scipy.ndimage import median_filter
from scipy.optimize import curve_fit
from tools.utils import *
from tools.ColorMaps import select_discrete_cmap


def get_bsqr_deviation_numerically(field, volts, N, threshold=3., savgol_num=501, savgol_poly=2,
                                   return_third_deriv=False, plot=False):
    # first numerically get the value where its much bigger than 2nd deriv avg

    first_deriv = savgol_filter(volts, savgol_num, savgol_poly, deriv=1)
    second_deriv = savgol_filter(first_deriv, savgol_num, savgol_poly, deriv=1)
    third_deriv = savgol_filter(second_deriv, savgol_num, 2, deriv=1)

    mean_2nd_deriv_og = np.mean(second_deriv[N:2 * N])
    mean_3rd_deriv_og = np.mean(third_deriv[N:2 * N])

    dev_locs = np.abs(second_deriv) > threshold * np.abs(mean_2nd_deriv_og)  # + 25 * std_2nd_deriv_og
    dev_locs3 = np.abs(third_deriv) > threshold * np.abs(mean_3rd_deriv_og)  # + 25 * std_2nd_deriv_og

    dev_loc2nd = np.argmax(dev_locs > 0)
    dev_loc3rd = np.argmax(dev_locs3 > 0)

    if plot:
        fig, axs = plt.subplots(ncols=2, figsize=(16, 9))
        axs[0].plot(field, volts, linewidth=2, c='indianred', label='Data')
        # axs[0].plot(field, func(field,alpha, beta, gamma),linewidth=2,c='midnightblue',label='Fit',linestyle='dashed')

        axs[1].plot(field, second_deriv, linewidth=2, c='darkgray')
        axs[1].axvline(field[dev_loc2nd])
        axs[0].axvline(field[dev_loc2nd])

        axs[0].legend(framealpha=0, ncol=1, loc='best',
                      prop={'size': 24, 'family': 'arial'})
        publication_plot(axs[0], 'Magnetic Field (T)', 'Torque (arb.)')
        publication_plot(axs[1], 'Magnetic Field (T)', r'$\frac{\partial^2 \tau}{\partial B^2}$')
        plt.tight_layout(pad=1)
        plt.show()

    if return_third_deriv:
        return field[dev_loc2nd], field[dev_loc3rd]
    else:
        return field[dev_loc2nd]


def get_bsqr_deviation_analytic(field, volts, N, plot=False, threshold=3., std=False):


    func = lambda x, alpha, beta, gamma, delta: alpha * x ** 2 + beta * np.exp(x * gamma) + delta

    # func = lambda x, alpha, beta, gamma : alpha * x ** 2 + beta / (x - gamma)

    popt, pcov = curve_fit(func, field, volts, maxfev=150000)

    alpha = popt[0]
    beta = popt[1]
    gamma = popt[2]
    delta = popt[3]

    err_in_fit = np.sqrt(np.diag(pcov))

    err_in_deriv = 2 * err_in_fit[0] + np.sqrt((err_in_fit[1] / beta) ** 2 + 2 * (err_in_fit[2] / gamma) ** 2)

    second_deriv = 2 * alpha + beta * gamma ** 2 * np.exp(gamma * field)

    mean_2nd_deriv_og = np.mean(second_deriv[:N])
    std_2nd_deriv_og = np.std(second_deriv[:N])

    locs_of_mean = np.arange(N)
    locs = np.setdiff1d(np.arange(len(field)), locs_of_mean)

    if not std:
        dev_locs = np.abs(second_deriv)[locs] > threshold * np.abs(mean_2nd_deriv_og)  # + 25 * std_2nd_deriv_og
    else:
        dev_locs = np.abs(second_deriv)[locs] > threshold * std_2nd_deriv_og

    min_err = np.abs(second_deriv)[locs] > threshold * np.abs(mean_2nd_deriv_og) - err_in_deriv
    max_err = np.abs(second_deriv)[locs] > threshold * np.abs(mean_2nd_deriv_og) + err_in_deriv

    dev_loc2nd = np.argmax(dev_locs > 0) + N

    min_err_loc = np.argmax(min_err > 0) + N
    max_err_loc = np.argmax(max_err > 0) + N

    if plot:
        fig, axs = plt.subplots(ncols=2, figsize=(16, 9))
        axs[0].plot(field, volts, linewidth=2, c='indianred', label='Data')
        axs[0].plot(field, func(field, alpha, beta, gamma, delta), linewidth=2, c='midnightblue', label='Fit',
                    linestyle='dashed')

        axs[1].plot(field, second_deriv, linewidth=2, c='darkgray')
        axs[1].axvline(field[dev_loc2nd])
        axs[0].axvline(field[dev_loc2nd])

        axs[0].legend(framealpha=0, ncol=1, loc='best',
                      prop={'size': 24, 'family': 'arial'})
        publication_plot(axs[0], 'Magnetic Field (T)', 'Torque (arb.)')
        publication_plot(axs[1], 'Magnetic Field (T)', r'$\frac{\partial^2 \tau}{\partial B^2}$')
        plt.tight_layout(pad=1)
        plt.show()

    return field[dev_loc2nd], field[min_err_loc], field[max_err_loc]


files = [
    '/Volumes/GoogleDrive/Shared drives/Quantum Materials - Sebastian Group/MagnetTime/2019_09_TLH/TLH_2019.09_HL_hybrid/VT7/angles/0.8K_-7deg_sweep178_up.csv',
    '/Volumes/GoogleDrive/Shared drives/Quantum Materials - Sebastian Group/MagnetTime/2019_09_TLH/TLH_2019.09_HL_hybrid/VT7/angles/0.3K_-3.5deg_sweep134_up.csv',
    '/Volumes/GoogleDrive/Shared drives/Quantum Materials - Sebastian Group/MagnetTime/2019_09_TLH/TLH_2019.09_HL_hybrid/VT7/angles/0.3K_-10.5deg_sweep152_up.csv',
    '/Volumes/GoogleDrive/Shared drives/Quantum Materials - Sebastian Group/MagnetTime/2019_09_TLH/TLH_2019.09_HL_hybrid/VT7/angles/0.3K_-14deg_sweep014_up.csv',
    '/Volumes/GoogleDrive/Shared drives/Quantum Materials - Sebastian Group/MagnetTime/2019_09_TLH/TLH_2019.09_HL_hybrid/VT7/angles/0.3K_-14deg_sweep157_up.csv',
    '/Volumes/GoogleDrive/Shared drives/Quantum Materials - Sebastian Group/MagnetTime/2019_09_TLH/TLH_2019.09_HL_hybrid/VT7/angles/0.3K_-42deg_sweep022_up.csv',
    '/Volumes/GoogleDrive/Shared drives/Quantum Materials - Sebastian Group/MagnetTime/2019_09_TLH/TLH_2019.09_HL_hybrid/VT7/angles/0.3K_-56deg_sweep025_up.csv',
    '/Volumes/GoogleDrive/Shared drives/Quantum Materials - Sebastian Group/MagnetTime/2019_09_TLH/TLH_2019.09_HL_hybrid/VT7/angles/0.3K_-84deg_sweep030_up.csv',
    '/Volumes/GoogleDrive/Shared drives/Quantum Materials - Sebastian Group/MagnetTime/2019_09_TLH/TLH_2019.09_HL_hybrid/VT7/angles/0.3K_0deg_sweep048_up.csv',
    '/Volumes/GoogleDrive/Shared drives/Quantum Materials - Sebastian Group/MagnetTime/2019_09_TLH/TLH_2019.09_HL_hybrid/VT7/angles/0.3K_3.5deg_sweep053_up.csv',
    '/Volumes/GoogleDrive/Shared drives/Quantum Materials - Sebastian Group/MagnetTime/2019_09_TLH/TLH_2019.09_HL_hybrid/VT7/angles/0.3K_7deg_sweep058_up.csv',
    '/Volumes/GoogleDrive/Shared drives/Quantum Materials - Sebastian Group/MagnetTime/2019_09_TLH/TLH_2019.09_HL_hybrid/VT7/angles/0.3K_10.5deg_sweep082_up.csv',
    '/Volumes/GoogleDrive/Shared drives/Quantum Materials - Sebastian Group/MagnetTime/2019_09_TLH/TLH_2019.09_HL_hybrid/VT7/angles/0.3K_14deg_sweep087_up.csv',
    '/Volumes/GoogleDrive/Shared drives/Quantum Materials - Sebastian Group/MagnetTime/2019_09_TLH/TLH_2019.09_HL_hybrid/VT7/angles/0.3K_17.5deg_sweep092_up.csv',
    '/Volumes/GoogleDrive/Shared drives/Quantum Materials - Sebastian Group/MagnetTime/2019_09_TLH/TLH_2019.09_HL_hybrid/VT7/angles/0.3K_21deg_sweep097_up.csv',
    '/Volumes/GoogleDrive/Shared drives/Quantum Materials - Sebastian Group/MagnetTime/2019_09_TLH/TLH_2019.09_HL_hybrid/VT7/angles/0.3K_24.5deg_sweep102_up.csv',
    '/Volumes/GoogleDrive/Shared drives/Quantum Materials - Sebastian Group/MagnetTime/2019_09_TLH/TLH_2019.09_HL_hybrid/VT7/angles/0.3K_28deg_sweep109_up.csv',
    '/Volumes/GoogleDrive/Shared drives/Quantum Materials - Sebastian Group/MagnetTime/2019_09_TLH/TLH_2019.09_HL_hybrid/VT7/angles/0.3K_31.5deg_sweep117_up.csv',
    '/Volumes/GoogleDrive/Shared drives/Quantum Materials - Sebastian Group/MagnetTime/2019_09_TLH/TLH_2019.09_HL_hybrid/VT7/angles/0.3K_35deg_sweep123_up.csv',
    '/Volumes/GoogleDrive/Shared drives/Quantum Materials - Sebastian Group/MagnetTime/2019_09_TLH/TLH_2019.09_HL_hybrid/VT7/angles/0.3K_38.5deg_sweep126_up.csv']

angles_lst = []

for f in files:
    actual_file_name = f.split('/')[-1]

    first_bit = actual_file_name.split('_')[1]

    angle = float(first_bit.split('d')[0])

    angles_lst.append(angle)

sorted_angle_inds = np.argsort(angles_lst)

angles_number = np.array(angles_lst)[sorted_angle_inds] - 17.5

print(angles_number)

files = np.array(files)[sorted_angle_inds]

cmap = select_discrete_cmap('venasaur')

all_angles, all_bdevs = [], []
all_min_error, all_max_error = [], []


fig2, ax_scat = MakePlot(figsize=(6, 6), ncols=1).create()

fig, a = MakePlot(figsize=(8, 8), gs=True).create()

gs = fig.add_gridspec(2, 2)
ax3 = fig.add_subplot(gs[:, 0])
ax4 = fig.add_subplot(gs[0, 1])
ax5 = fig.add_subplot(gs[1, 1])

med_num, savgol_num = 51, 501
N = 50
threshold = 1.001
threshold_numeric = 3
for i, file in enumerate(files):

    B = np.genfromtxt(file, delimiter=',', skip_header=4)[:, 0]
    flip = False
    if B[0] > B[-1]:
        flip = True
        B = np.flip(B)
    B_copy = B
    B_inds1 = B > 1
    B_inds2 = B < 32
    B_inds = B_inds1 & B_inds2
    # B = B[B_inds]
    x = B

    # tau = savgol_filter(median_filter(1e3*np.genfromtxt(file, delimiter='\t', skip_header=4)[:,1],med_num),savgol_num,5)

    tau = median_filter(1e3 * np.genfromtxt(file, delimiter=',', skip_header=4)[:, 1], med_num)
    if flip:
        tau = np.flip(tau)

    tau_copy = tau

    # tau = tau[B_inds]

    field = B
    volts = tau

    # volts -= volts[0]

    if volts[0] > volts[10]:
        volts *= -1

    B_dev_num = get_bsqr_deviation_numerically(field, volts, 500, threshold=threshold_numeric, plot=False)
    B_dev_anal, min_err_loc, max_err_loc = get_bsqr_deviation_analytic(field, volts, 500, plot=False,
                                                                       threshold=threshold)

    # ax1.scatter(angles_number[i], B[dev_loc])

    ax3.plot(B, tau - tau[0], linewidth=2, c=plt.cm.autumn(i / len(files)),
             label=str(angles_number[i]) + '$^\mathregular{o}$', alpha=.6)

    ax4.scatter(angles_number[i], B_dev_num, s=200, c=plt.cm.autumn(i / len(files)), alpha=.4)
    ax5.scatter(angles_number[i], B_dev_anal, s=200, c=plt.cm.autumn(i / len(files)), alpha=.4)
    ax_scat.scatter(angles_number[i], B_dev_anal, s=200, c='indianred', alpha=.9)

    all_angles.append(angles_number[i])
    all_bdevs.append(B_dev_anal)

    all_min_error.append(min_err_loc)
    all_max_error.append(max_err_loc)

# ax4.set_xticks([0, 20, 40, 60, 80, 100])
# ax4.set_xticklabels([0, 20, 40, '', 80, 100])
#
# ax5.set_xticks([0, 20, 40, 60, 80, 100])
# ax5.set_xticklabels([0, 20, 40, '', 80, 100])

sorted_inds = np.argsort(all_angles)

all_angles_array = np.array(all_angles)[sorted_inds]
all_bdevs_array = np.array(all_bdevs)[sorted_inds]

all_min = np.array(all_min_error)[sorted_inds]
all_max = np.array(all_max_error)[sorted_inds]

# yerr = np.array([all_min, all_max])

print(all_angles_array)
print(all_max)
print(all_min)
yerr = all_max / 2
# yerr /=2

# yerr = all_min
fit = np.poly1d(np.polyfit(all_angles_array, all_bdevs_array, 4))

linspace = np.linspace(0, 98)

ax_scat.plot(linspace, fit(linspace), linewidth=2, c='darkslategray')
ax_scat.errorbar(all_angles_array, all_bdevs_array, yerr=yerr, fmt='none', c='k', linewidth=2.1)

#
# df = pd.DataFrame(np.array([all_angles_array, all_bdevs_array, yerr]).T,columns=['Angle', 'B_dev', 'yerr'])
#
# df.to_csv(main_path+'B2_dev.csv', index=False)

publication_plot(ax3, 'Magnetic Field (T)', 'Torque (arb.)')
publication_plot(ax4, r'$\theta$', '$B_{\mathrm{Numerical}}$')
publication_plot(ax5, r'$\theta$', '$B_{\mathrm{Analytical}}$')
publication_plot(ax_scat, r'$\theta$', '$B_{\mathrm{Analytical}}$')
plt.tight_layout(pad=1.5)

legend = ax3.legend(framealpha=0, ncol=2, loc='best',
                    prop={'size': 24, 'family': 'arial'}, handlelength=0, columnspacing=0.5)

for line, text in zip(legend.get_lines(), legend.get_texts()):
    text.set_color(line.get_color())

plt.tight_layout(pad=.8)
plt.show()





