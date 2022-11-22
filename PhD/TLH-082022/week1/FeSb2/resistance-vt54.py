import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter, argrelmin, argrelmax
from scipy.interpolate import interp1d
from scipy.ndimage import median_filter
from tools.utils import *
from tools.ColorMaps import select_discrete_cmap
from tools.utils import *
from tools.ColorMaps import *
from scipy.optimize import curve_fit
from scipy.signal import medfilt
import sounddevice as sd


def get_bsqr_deviation_analytic_no_const(field, volts, N, plot=False,threshold=3.,std=False):

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



colours = ['#001219',
           '#005F73',
           '#0A9396',
           '#94D2BD',
           '#E9D8A6',
           '#EE9B00',
           '#CA6702',
           '#BB3E03',
           '#AE2012',
           '#9B2226']

resistance_path = '/Volumes/GoogleDrive/Shared drives/Quantum Materials/MagnetTime/2022_08_TLH/week1/FeSb2-VT54/resistance/'
# torque_path = '/Volumes/GoogleDrive/Shared drives/Quantum Materials/MagnetTime/2022_08_TLH/week1/FeSb2-VT19/torque/'

fig, a = MakePlot(figsize=(12, 8), gs=True).create()
gs = fig.add_gridspec(2,1)

ax1 = fig.add_subplot(gs[0,0])
ax2 = fig.add_subplot(gs[1,0])


sweeps = ['0.29K_103.33deg_sweep094_up.csv',
          '0.29K_96.66deg_sweep096_up.csv',
          '0.29K_90deg_sweep099_up.csv',
          '0.29K_83.33deg_sweep101_up.csv',
          '0.29K_76.66deg_sweep103_up.csv',
          '0.29K_70deg_sweep107_up.csv',
          '0.29K_63.33deg_sweep110_down.csv']

lstyles = ['solid', 'dashed']

angles = [103.33, 96.66, 90, 83.33, 76.66, 70, 63.33]

med_num, savgol_num = 51, 501
N=50
threshold=1.5

grads = []

for i, s_name in enumerate(sweeps):


    resistance_dat = load_matrix(resistance_path + s_name)
    B = resistance_dat[:, 0]
    V = resistance_dat[:, 1]  # mV

    I = 10e-6 #A

    R = np.abs(V / I) #Ohm, I tink abs val

    interp_B = np.linspace(np.min(B), np.max(B), 10000)
    interp_R = np.interp(interp_B, B, R)

    deriv_R = savgol_filter(interp_R,111,3,1)


    ax1.plot(B, R, linewidth=2, c=colours[i], label=str(angles[i])+r'$\degree$')
    ax2.plot(interp_B, deriv_R, linewidth=2, c=colours[i], label=str(angles[i]) + r'$\degree$')

publication_plot(ax1, r'$B$ (T)', r'$R$ ($\Omega$)')
publication_plot(ax2, r'$B$ (T)', r'$\frac{\partial R}{\partial B}$ ($\Omega$T$^-1$)')


handles, labels = ax1.get_legend_handles_labels()

# ax1.set_ybound(-10,18)

ax1.set_xbound(0,40)

ax2.set_ybound(-0.01,0.01)


legend = ax1.legend(handles, labels, framealpha=0, ncol=1, loc='lower right',
                    prop={'size': 18, 'family': 'arial'},
                    handlelength=0)

for line,text in zip(legend.get_lines(), legend.get_texts()):
    text.set_color(line.get_color())

plt.tight_layout(pad=1)

plt.show()



