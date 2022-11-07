import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.signal import savgol_filter, argrelmin, argrelmax
from scipy.ndimage import median_filter
from scipy.optimize import curve_fit
from tools.utils import *
import matplotlib
from tools.ColorMaps import select_discrete_cmap


def get_bsqr_deviation_numerically(field, volts, N, threshold=3, savgol_num=501, savgol_poly=2,
                                   return_third_deriv=False):
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

    if return_third_deriv:
        return field[dev_loc2nd], field[dev_loc3rd]
    else:
        return field[dev_loc2nd]


def get_bsqr_deviation_analytic(field, volts, N, plot=False, threshold=3.):
    func = lambda x, alpha, beta, gamma, delta: alpha * x ** 2 + beta * np.exp(x * gamma) + delta

    # func = lambda x, alpha, beta, gamma : alpha * x ** 2 + beta / (x - gamma)

    popt, pcov = curve_fit(func, field, volts, p0=(1, .0001, 1, 1), maxfev=15000)  # 10000,0.0001,0.000000001
    # best p0 is (1,.0001,1) so far
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

    err_in_deriv = 2 * err_in_fit[0] + np.sqrt((err_in_fit[1] / beta) ** 2 + 2 * (err_in_fit[2] / gamma) ** 2)

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


med_num, savgol_num = 51, 501
N=50
threshold=1.5

B = np.genfromtxt('/Users/npopiel/Desktop/Hybrid/VT15-hybrid_21.dat', delimiter='\t', skip_header=4)[:, 0]
B_copy = B
B_inds1 = B > 8
B_inds2 = B < 46
B_inds = B_inds1 & B_inds2
B = B[B_inds]
x = B

# tau = savgol_filter(median_filter(1e3*np.genfromtxt(file, delimiter='\t', skip_header=4)[:,1],med_num),savgol_num,5)

tau = median_filter(828 * np.genfromtxt('/Users/npopiel/Desktop/Hybrid/VT15-hybrid_21.dat'
                                        , delimiter='\t', skip_header=4)[:, 1], med_num)

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

func = lambda x, alpha, beta, gamma, delta: alpha * x ** 2 + beta * np.exp(x * gamma) + delta

# func = lambda x, alpha, beta, gamma : alpha * x ** 2 + beta / (x - gamma)

popt, pcov = curve_fit(func, field, volts, p0=(1, .0001, 1, 1), maxfev=15000)  # 10000,0.0001,0.000000001
# best p0 is (1,.0001,1) so far
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

err_in_deriv = 2 * err_in_fit[0] + np.sqrt((err_in_fit[1] / beta) ** 2 + 2 * (err_in_fit[2] / gamma) ** 2)

min_err = np.abs(second_deriv)[locs] > threshold * np.abs(mean_2nd_deriv_og) - err_in_deriv
max_err = np.abs(second_deriv)[locs] > threshold * np.abs(mean_2nd_deriv_og) + err_in_deriv

dev_loc2nd = np.argmax(dev_locs > 0) + N

min_err_loc = np.argmax(min_err > 0) + N
max_err_loc = np.argmax(max_err > 0) + N

fig, ax = MakePlot(ncols=1, figsize=(8, 9)).create()

ax.plot(field, volts, linewidth=2, c='k', label='Data')
ax.plot(field, func(field, alpha, beta, gamma, delta), linewidth=8, c='#BF7CDC', label='Fit',
            alpha=0.4)
ax.plot(field, alpha * field ** 2 + delta, linewidth=2, c='#982649', label='Quadratic', alpha=.8
)

# ax.plot(field, beta * np.exp(gamma * field), linewidth=3, c='#60B2E5', label='Exponential', alpha=0.6
# )

exp = beta * np.exp(gamma * field)

# axs[0].plot(field, exp + (volts[-1] - exp[-1]) , linewidth=3, c='#011936', label='Exponential',
#             linestyle='dashed')

# axs[1].plot(field, second_deriv, linewidth=2, c='darkgray')
# axs[1].axvline(field[dev_loc2nd],c='#0D1B2A')
# axs[1].fill_between(field[:dev_loc2nd],1.5*mean_2nd_deriv_og, color='indianred',alpha=.3)
ax.axvline(field[dev_loc2nd], c='grey',linestyle='dashed')

bbox_args = dict(boxstyle="round", fc="0.8")
arrow_args = dict(width=0.8, headlength=10, headwidth=8,color='k')
trans = ax.get_xaxis_transform()
ax.annotate(r'$(\mu_0H)^2$' + '\nDeviation', xy=(field[dev_loc2nd], 8), xycoords='data',
                 xytext=(15.5, 6),
                 ha="center", va="center",
                 arrowprops=arrow_args, fontname='arial', fontsize=22)

axins = fig.add_axes([0.53, 0.62, 0.27, 0.32])

axins.plot(field ** 2, volts, linewidth=1, c='k')
axins.plot(field ** 2, func(field, alpha, beta, gamma, delta), linewidth=8, c='#BF7CDC',
            alpha=.4)
axins.plot(field ** 2, alpha * field ** 2 + delta, linewidth=2, c='#982649', alpha=.8
)

axins.axvline(field[dev_loc2nd] ** 2, c='grey',linestyle='dashed')


publication_plot(axins, r'$(\mu_0H)^2$ (T$^2$)', r'$\tau$  ($\times 10^{-2}$ $\mu_B$ T per f.u.)', label_fontsize=14, tick_fontsize=12)

# ax.annotate(r'Data', xy=(35, 6.97), xycoords='data',
#                  xytext=(30, 6.97),
#                  ha="center", va="center",
#                  arrowprops=arrow_args, fontname='arial', fontsize=22)

#
# ax.annotate(r'$\textcolor{#BF7CDC}{\mathrm{Fit}} = \textcolor{#982649}{\mathrm{Quadratic}} + \textcolor{#60B2E5}{\mathrm{Exponential}}$', xy=(field[dev_loc2nd], 11), xycoords='data',
#                  xytext=(15, 12.5),
#                  ha="left", va="center",
#                  arrowprops=arrow_args, fontname='arial', fontsize=22)



publication_plot(ax, r' $\mu_0H $ (T)', r'$\tau$  ($\times 10^{-2}$ $\mu_B$ T per f.u.)')


legend = ax.legend(framealpha=0, ncol=1, loc='upper left',
              prop={'size': 24, 'family': 'arial'}, handlelength=0, columnspacing = .5)

for line,text in zip(legend.get_lines(), legend.get_texts()):
    text.set_color(line.get_color())

for item in legend.legendHandles:
    item.set_visible(False)

ax.annotate(r'$\phi$ = 21$\degree$', xy=(43, 0), xycoords='data',
                 ha="center", va="center", color='k',
                 fontname='arial', fontsize=26)

plt.tight_layout(pad=1)
# plt.show()
fig.savefig('/Users/npopiel/Desktop/Hybrid/'+'torqueFit-fixed.pdf', bbox_inches='tight')


