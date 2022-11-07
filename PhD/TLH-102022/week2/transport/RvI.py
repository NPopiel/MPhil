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

def load_xy(fileandpath, filenumber, up=True,y_root='Coil_X_', x_root = 'Field_', skiprows=9, space_x=False, space_y=False):
    dat  = pd.read_csv(fileandpath, delimiter='\t', skiprows=skiprows)

    if space_x:
        x = np.array(dat[x_root + str(filenumber) + ' '])
    else:
        x = np.array(dat[x_root + str(filenumber)])

    if space_y:
        y = np.array(dat[y_root + str(filenumber) + ' '])
    else:
        y = np.array(dat[y_root + str(filenumber)])

    if not up:
        x = np.flip(x)
        y = np.flip(y)
    return x, y

resistance_path = '/Volumes/GoogleDrive/Shared drives/Quantum Materials - Sebastian Group/MagnetTime/2022_10_TLH/week2/'

# torque_path = '/Volumes/GoogleDrive/Shared drives/Quantum Materials/MagnetTime/2022_08_TLH/week1/FeSb2-VT19/torque/'

fig, a = MakePlot(figsize=(12, 8), gs=True).create()
gs = fig.add_gridspec(1,2)

ax1 = fig.add_subplot(gs[0,0])
ax2 = fig.add_subplot(gs[0,1])

Vxx_root = 'Vxx/day2/'
Vxy_root = 'Vxy/day2/'

file_root = 'Cambridge_October.'
file_end = '.txt'

filenumbers_up = ['063', '066', '071', '075', '078', '084', '088', '091', '094']

filenumbers_dn = ['064', '067', '073', '076', '079', '086', '089', '092', '095']

lstyles = ['solid', 'dashed']


Is = [1e-6]

med_num, savgol_num = 51, 501
N=50
threshold=1.5

grads = []

field_root = 'Field_'
Volts_xx_root = 'VT166_Rxx_R_'
Volts_xy_root = 'VT166_Rxy_R_'

for i, s_num in enumerate(filenumbers_up):


    B, Vxx = load_xy(resistance_path+file_root+str(s_num)+file_end, str(s_num),x_root=field_root,
                     y_root=Volts_xx_root,skiprows=9)
    _, Vxy = load_xy(resistance_path + file_root + str(s_num) + file_end, str(s_num), x_root=field_root,
                     y_root=Volts_xy_root, skiprows=9)

    locs1 = B > 0
    locs2 = B < 45

    locs = locs1 & locs2

    B = B[locs]
    Vxx = Vxx[locs]
    Vxy = Vxy[locs]

    I = Is[i]

    Rxx = Vxx / I
    Rxy = Vxy / I

    ax1.plot(B, Rxx, linewidth=2, c=colours[i], label=str(Is[i])+r' A')
    ax2.plot(B, Rxy, linewidth=2, c=colours[i], label=str(Is[i])+r' A')

    B, Vxx = load_xy(resistance_path + file_root + filenumbers_dn[i] + file_end, str(filenumbers_dn[i]), x_root=field_root,
                     y_root=Volts_xx_root, skiprows=9)
    _, Vxy = load_xy(resistance_path + file_root + filenumbers_dn[i] + file_end, filenumbers_dn[i], x_root=field_root,
                     y_root=Volts_xy_root, skiprows=9)

    B = B[locs]
    Vxx = Vxx[locs]
    Vxy = Vxy[locs]

    Rxx = Vxx / I
    Rxy = Vxy / I

    ax1.plot(B, Rxx, linewidth=2, c=colours[i], linestyle='dashed')
    ax2.plot(B, Rxy, linewidth=2, c=colours[i], linestyle='dashed')

publication_plot(ax1, r'$B$ (T)', r'$R_{xx}$ ($\Omega$)')
publication_plot(ax2, r'$B$ (T)', r'$R_{xy}$ ($\Omega$)')


handles, labels = ax1.get_legend_handles_labels()

# ax1.set_ybound(-10,18)

ax1.set_xbound(0,42)
ax2.set_xbound(0,42)

ax1.annotate(r'$T = 2.5$ K', xy=(0.3, 0.9), xycoords='axes fraction',
              ha="left", va="center", fontname='arial', fontsize=22)

# ax1.set_ybound(0,9e3)
# ax2.set_ybound(0,9e3)

# ax2.set_ybound(-0.01,0.01)


legend = ax1.legend(handles, labels, framealpha=0, ncol=1, loc='lower right',
                    prop={'size': 18, 'family': 'arial'},
                    handlelength=0)

for line,text in zip(legend.get_lines(), legend.get_texts()):
    text.set_color(line.get_color())

plt.tight_layout(pad=1)

plt.show()




