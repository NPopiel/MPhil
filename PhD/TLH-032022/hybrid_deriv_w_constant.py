import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.signal import savgol_filter, argrelmin, argrelmax
from scipy.ndimage import median_filter
from scipy.optimize import curve_fit
from tools.utils import *
from tools.ColorMaps import select_discrete_cmap



def get_bsqr_deviation_numerically(field, volts, N, threshold=3,savgol_num=501,savgol_poly=2, return_third_deriv=False):

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

    if return_third_deriv:
        return field[dev_loc2nd], field[dev_loc3rd]
    else:
        return field[dev_loc2nd]


def get_bsqr_deviation_analytic(field, volts, N, plot=False,threshold=3.):

    func = lambda x, alpha, beta, gamma, delta : alpha * x ** 2 + beta * np.exp(x * gamma) + delta

    # func = lambda x, alpha, beta, gamma : alpha * x ** 2 + beta / (x - gamma)

    popt, pcov = curve_fit(func, field, volts,p0=(1,.0001,1,1),maxfev=15000) #10000,0.0001,0.000000001
    #best p0 is (1,.0001,1) so far
    # ( 9.254070896283006e-05 , -0.003231388939997808 , 0.16450428722133764 ) is fine

    alpha = popt[0]
    beta = popt[1]
    gamma = popt[2]
    delta = popt[3]

    locs_of_mean = np.arange(N)
    locs = np.setdiff1d(np.arange(len(field)), locs_of_mean)

    second_deriv = 2 * alpha + beta * gamma ** 2 * np.exp(gamma * field)

    mean_2nd_deriv_og = np.mean(second_deriv[:N])

    dev_locs = np.abs(second_deriv)[locs] > threshold * np.abs(mean_2nd_deriv_og)  # + 25 * std_2nd_deriv_og

    err_in_fit = np.sqrt(np.diag(pcov))

    err_in_deriv = 2*err_in_fit[0] + np.sqrt((err_in_fit[1]/beta)**2 + 2*(err_in_fit[2]/gamma)**2)

    min_err =  np.abs(second_deriv)[locs] > threshold * np.abs(mean_2nd_deriv_og) - err_in_deriv
    max_err =  np.abs(second_deriv)[locs] > threshold * np.abs(mean_2nd_deriv_og) + err_in_deriv

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




files = [#'/Users/npopiel/Desktop/Hybrid/VT15-hybrid_0.dat',
'/Users/npopiel/Desktop/Hybrid/VT15-hybrid_7.dat',
'/Users/npopiel/Desktop/Hybrid/VT15-hybrid_10p5.dat',
'/Users/npopiel/Desktop/Hybrid/VT15-hybrid_12.dat',
'/Users/npopiel/Desktop/Hybrid/VT15-hybrid_14.dat',
'/Users/npopiel/Desktop/Hybrid/VT15-hybrid_15p5.dat',
'/Users/npopiel/Desktop/Hybrid/VT15-hybrid_16p5.dat',
'/Users/npopiel/Desktop/Hybrid/VT15-hybrid_17.dat',
'/Users/npopiel/Desktop/Hybrid/VT15-hybrid_21.dat']

angles = ['7$^\mathregular{o}$', '10.5$^\mathregular{o}$', '12$^\mathregular{o}$', '14$^\mathregular{o}$',
               '15.5$^\mathregular{o}$', '16.5$^\mathregular{o}$', '17$^\mathregular{o}$', '21$^\mathregular{o}$']

angles_number = [7, 10.5, 12, 14, 15.5, 16.5, 17, 21]




fig, a = MakePlot(figsize=(16,8), gs=True).create()

gs = fig.add_gridspec(2,3)
ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[1, 0])
ax3 = fig.add_subplot(gs[:, 1])
ax4 = fig.add_subplot(gs[:, 2])

fig2, ax_scat = MakePlot(figsize=(6,6),ncols=1).create()



all_bdevs, all_angles = [], []
all_min_error, all_max_error = [], []

med_num, savgol_num = 51, 501
N=50
threshold=1.5
for i, file in enumerate(files):

    B = np.genfromtxt(file, delimiter='\t', skip_header=4)[:,0]
    B_copy = B
    B_inds1 = B > 8
    B_inds2 = B < 46
    B_inds = B_inds1 & B_inds2
    B = B[B_inds]
    x=B

    # tau = savgol_filter(median_filter(1e3*np.genfromtxt(file, delimiter='\t', skip_header=4)[:,1],med_num),savgol_num,5)

    tau = median_filter(1e3 * np.genfromtxt(file, delimiter='\t', skip_header=4)[:, 1], med_num)

    # Only do this if angle 0 is in!!

    # if i == 0:
    #     tau *=-1

    tau_copy = tau

    tau = tau[B_inds]

    field = B
    volts = tau

    volts -= volts[0]

    if volts[0] > volts[10]:

        volts *= -1


    B_dev_num = get_bsqr_deviation_numerically(field, volts, 500,threshold=4)
    B_dev_anal, min_err, max_err = get_bsqr_deviation_analytic(field, volts, 500,plot=True,threshold=threshold)

    # ax1.scatter(angles_number[i], B[dev_loc])

    ax3.plot(B, tau - tau[0], linewidth=2, c=plt.cm.jet(i/len(files)), label=str(angles[i]), alpha=.6)
    # ax4.plot(B ** 2, tau - tau[0], linewidth=2, c=plt.cm.jet(i / len(files)), label=str(angles[i]), alpha=.6)

    ax1.scatter(angles_number[i], B_dev_num, s=200, c=plt.cm.jet(i/len(files)), alpha=.4)
    ax2.scatter(angles_number[i], B_dev_anal, s=200, c=plt.cm.jet(i/len(files)), alpha=.4)
    ax_scat.scatter(angles_number[i], B_dev_anal, s=200, c='indianred', alpha=.9)

    all_angles.append(angles_number[i])
    all_bdevs.append(B_dev_anal)

    all_min_error.append(min_err)
    all_max_error.append(max_err)

sorted_inds = np.argsort(all_angles)

all_angles_array = np.array(all_angles)[sorted_inds]
all_bdevs_array = np.array(all_bdevs)[sorted_inds]

all_min = np.array(all_min_error)[sorted_inds]
all_max = np.array(all_max_error)[sorted_inds]


df = pd.DataFrame(np.array([all_angles_array, all_bdevs_array, all_max/2]).T, columns=['Angle', 'B_dev', 'yerr'])

df.to_csv(r'/Users/npopiel/Desktop/Hybrid/B2_dev_c-aVT15.csv', index=False)


all_angles, all_bdevs = [], []
all_min_error, all_max_error = [], []


main_path = '/Users/npopiel/Desktop/Hybrid/'

files = [
'/Users/npopiel/Desktop/Hybrid/VT4-cell12_0.dat',
'/Users/npopiel/Desktop/Hybrid/VT4-cell12_7.dat',
'/Users/npopiel/Desktop/Hybrid/VT4-cell12_14.dat',
'/Users/npopiel/Desktop/Hybrid/VT4-cell12_21.dat',
'/Users/npopiel/Desktop/Hybrid/VT4-cell12_28.dat']

angles = ['0$^\mathregular{o}$', '7$^\mathregular{o}$',  '14$^\mathregular{o}$','21$^\mathregular{o}$', '28$^\mathregular{o}$']

angles_number = [0, 7, 14,21, 28]

upper_Bs = 45*np.ones(len(angles_number))#[35,30,35,35]#



for i, file in enumerate(files):

    B = np.genfromtxt(file, delimiter='\t', skip_header=4)[:,0]
    B_copy = B
    B_inds1 = B > 17.6 # Changed this!
    B_inds2 = B < upper_Bs[i]
    B_inds = B_inds1 & B_inds2
    B = B[B_inds]
    x=B

    # tau = savgol_filter(median_filter(1e3*np.genfromtxt(file, delimiter='\t', skip_header=4)[:,1],med_num),savgol_num,5)

    tau = median_filter(1e3 * np.genfromtxt(file, delimiter='\t', skip_header=4)[:, 1], med_num)

    tau_copy = tau

    tau = tau[B_inds]

    field = B
    volts = tau

    volts -= volts[0]

    if volts[0] > volts[10]:

        volts *= -1


    B_dev_num = get_bsqr_deviation_numerically(field, volts, 5000,threshold=4)
    B_dev_anal, min_err, max_err = get_bsqr_deviation_analytic(field, volts, 5000,plot=False, threshold=threshold)

    print(angles_number[i])
    print(B_dev_anal)

    # ax1.scatter(angles_number[i], B[dev_loc])

    ax4.plot(B, tau - tau[0], linewidth=2, c=plt.cm.viridis(i/len(files)), label=str(angles[i]), alpha=.6)
    # ax4.plot(B ** 2, tau - tau[0], linewidth=2, c=plt.cm.viridis(i / len(files)), label=str(angles[i]), alpha=.6)

    ax1.scatter(angles_number[i], B_dev_num, s=200, c=plt.cm.viridis(i/len(files)), alpha=.4)
    ax2.scatter(angles_number[i], B_dev_anal, s=200, c=plt.cm.viridis(i/len(files)), alpha=.4)
    ax_scat.scatter(angles_number[i], B_dev_anal, s=200, c='indianred', alpha=.9)

    all_angles.append(angles_number[i])
    all_bdevs.append(B_dev_anal)

    all_min_error.append(min_err)
    all_max_error.append(max_err)



ax2.set_ylim(0,40)
ax1.ticklabel_format(useOffset=False)
publication_plot(ax3, 'Magnetic Field (T)', 'Torque (arb.)', title='VT15 Hybrid')
publication_plot(ax4, 'Magnetic Field (T)', 'Torque (arb.)', title='VT4 Cell 12')
publication_plot(ax1, r'$\phi$', '$B_{\mathrm{Numerical}}$',)
publication_plot(ax2, r'$\phi$', '$B_{\mathrm{Analytical}}$')

legend = ax3.legend(framealpha=0, ncol=2, loc='upper left',
              prop={'size': 24, 'family': 'arial'}, handlelength=0, columnspacing = .5)

for line,text in zip(legend.get_lines(), legend.get_texts()):
    text.set_color(line.get_color())

legend = ax4.legend(framealpha=0, ncol=2, loc='upper left',
              prop={'size': 24, 'family': 'arial'}, handlelength=0, columnspacing = .5)

for line,text in zip(legend.get_lines(), legend.get_texts()):
    text.set_color(line.get_color())


sorted_inds = np.argsort(all_angles)
all_angles_array = np.array(all_angles)[sorted_inds]
all_bdevs_array = np.array(all_bdevs)[sorted_inds]

all_min = np.array(all_min_error)[sorted_inds]
all_max = np.array(all_max_error)[sorted_inds]


yerr = all_max/2
# yerr /=2

# yerr = all_min
fit = np.poly1d(np.polyfit(all_angles_array, all_bdevs_array, 4))

linspace = np.linspace(0,35)

ax_scat.plot(linspace,fit(linspace), linewidth=2,c='darkslategray')
ax_scat.errorbar(all_angles_array, all_bdevs_array, yerr=yerr, fmt='none', c='k', linewidth=2.1)
publication_plot(ax_scat, r'$\phi$', '$B_{\mathrm{Analytical}}$')

df = pd.DataFrame(np.array([all_angles_array, all_bdevs_array,yerr]).T,columns=['Angle', 'B_dev','yerr'])
df.to_csv(main_path+'B2_dev-c-abVT4b.csv', index=False)


plt.tight_layout(pad=.8)
plt.show()