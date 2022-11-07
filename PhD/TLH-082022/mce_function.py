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

def plot_torque(filepath, filenumber, tau_root='Vx_VT16_Cap_', field_root = 'Field_'):
    dat = pd.read_csv(filepath, delimiter='\t', skiprows=9)

    T = np.array(dat[tau_root + str(filenumber)])
    B = np.array(dat[field_root + str(filenumber)])

    fig, ax = MakePlot(figsize=(8,12)).create()

    ax.plot(B, T/np.max(T), linewidth=2, c='indianred')

    publication_plot(ax, 'Magnetic Field (T)', 'Torque (arb.)')

    plt.tight_layout(pad=1)

    plt.show()

def load_torque(filepath, filenumber, upper_threshold=41, lower_threshold=1,tau_root='Vx_VT16_Cap_', field_root = 'Field_'):
    dat = pd.read_csv(filepath, delimiter='\t', skiprows=9)

    T = np.array(dat[tau_root + str(filenumber)])
    B = np.array(dat[field_root + str(filenumber)])

    locs1 = B > lower_threshold
    locs2 = B < upper_threshold

    locs = locs1 & locs2

    T = T[locs]
    B = B[locs]

    return B, T

def load_temps(fileandpath, filenumber, upper_threshold=41, lower_threshold=1, up=True,RuOx_root='Ruthox_Temp_', field_root = 'Field_'):
    dat  = pd.read_csv(fileandpath, delimiter='\t', skiprows=9)

    T = np.array(dat[RuOx_root + str(filenumber)])
    B = np.array(dat[field_root + str(filenumber)])

    locs1 = B > lower_threshold
    locs2 = B < upper_threshold

    locs = locs1 & locs2

    T = T[locs]
    B = B[locs]

    if not up:
        T = np.flip(T)
        B = np.flip(B)
    return T, B

def interp_and_math(T_up, B_up, T_down, B_down, num_pts = 20000):

    if np.max(B_up) > np.max(B_down):
        max_B = np.max(B_down)
    else:
        max_B = np.max(B_up)

    B_big_up = np.linspace(0,max_B, num_pts)
    interpd_T_up = np.interp(B_big_up, B_up, T_up)

    B_big_down = np.linspace(0,max_B, num_pts)
    interpd_T_down = np.interp(B_big_down, B_down, T_down)

    diff = np.abs(interpd_T_up - interpd_T_down)

    sum_T = interpd_T_up + interpd_T_down

    return B_big_up, interpd_T_up, B_big_down, interpd_T_down , diff, sum_T

def plot_MCE(B_up, T_up,tau_up, B_down, T_down, tau_down, B_big_up, diff, sum_T):

    fig, a = MakePlot(gs=True, figsize=(16,12)).create()

    gs = fig.add_gridspec(2, 2)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[1, 0])
    ax3 = fig.add_subplot(gs[0, 1])
    ax4 = fig.add_subplot(gs[1, 1])

    ax1.plot(B_up, tau_up, label='Up', c='indianred')
    ax1.plot(B_down, tau_down, label='Down', c='midnightblue')
    ax2.plot(B_up, T_up, label='Up', c='indianred')
    ax2.plot(B_down, T_down, label='Down', c='midnightblue')
    ax3.plot(B_big_up, diff, label='Difference',c='darkslategray')
    ax4.plot(B_big_up, sum_T, label='Sum',c='darkslategray')

    # ax.legend()
    publication_plot(ax1, 'Magnetic Field (T)', r'$\tau$ (arb.)')
    publication_plot(ax2, 'Magnetic Field (T)', r'$T$')
    publication_plot(ax3, 'Magnetic Field (T)', r'$\Delta T$')
    publication_plot(ax4, 'Magnetic Field (T)', r'$\Sigma T$')

    handles, labels = ax1.get_legend_handles_labels()

    legend = ax1.legend(handles, labels, framealpha=0,  loc='best',
                        prop={'size': 18, 'family': 'arial'},
                        handlelength=0, labelspacing=2.6)

    for line,text in zip(legend.get_lines(), legend.get_texts()):
        text.set_color(line.get_color())

    handles, labels = ax4.get_legend_handles_labels()

    legend = ax4.legend(handles, labels, framealpha=0,  loc='best',
                        prop={'size': 18, 'family': 'arial'},
                        handlelength=0, labelspacing=2.6)

    for line,text in zip(legend.get_lines(), legend.get_texts()):
        text.set_color(line.get_color())

    handles, labels = ax2.get_legend_handles_labels()

    legend = ax2.legend(handles, labels, framealpha=0,  loc='best',
                        prop={'size': 18, 'family': 'arial'},
                        handlelength=0, labelspacing=2.6)

    for line,text in zip(legend.get_lines(), legend.get_texts()):
        text.set_color(line.get_color())


    handles, labels = ax3.get_legend_handles_labels()

    legend = ax3.legend(handles, labels, framealpha=0,  loc='best',
                        prop={'size': 18, 'family': 'arial'},
                        handlelength=0, labelspacing=2.6)

    for line,text in zip(legend.get_lines(), legend.get_texts()):
        text.set_color(line.get_color())

    plt.tight_layout(pad=1)
    plt.show()

main_path = '/Volumes/GoogleDrive/Shared drives/Quantum Materials/MagnetTime/2022_08_TLH/week1/All Sweeps/'

fileandpath_up = main_path + 'Cambridge_August.156.txt'

fileandpath_down = main_path + 'Cambridge_August.157.txt'

plot_torque(fileandpath_up, '156')

T_up, B_up = load_temps(fileandpath_up,'156')
T_down, B_down = load_temps(fileandpath_down,'157',up=False)
B_ta, tau_up = load_torque(fileandpath_up,'156')
B, tau_down = load_torque(fileandpath_down,'157')

B_big_up, interpd_T_up, B_big_down, interpd_T_down , diff, sum_T = interp_and_math(T_up, B_up, T_down, B_down)

plot_MCE(B_up, T_up, tau_up, B_down, T_down, tau_down, B_big_up, diff, sum_T)