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



def load_torque(filepath, filenumber, upper_threshold=41.4, lower_threshold=.1,tau_root='Vx_VT16_Cap_', field_root = 'Field_', skiprows=9):
    dat = pd.read_csv(filepath, delimiter='\t', skiprows=skiprows)

    T = np.array(dat[tau_root + str(filenumber)])
    B = np.array(dat[field_root + str(filenumber)])

    locs1 = B > lower_threshold
    locs2 = B < upper_threshold

    locs = locs1 & locs2

    T = T[locs]
    B = B[locs]

    if B[0] > B[-1]:
        B = np.flip(B)
        T = np.flip(T)

    T -= T[0]

    return B, T


colours = ['#5F0F40',
           '#6E0C38',
           '#7D092F',
           '#9A031E',
           '#B32520',
           '#CB4721',
           '#FB8B24',
           '#E36414',
           '#AE5E26',
           '#795838',
           '#44524A',
           '#2A4F53',
           '#0F4C5C',
           '#255C6B']


main_path = '/Volumes/GoogleDrive/Shared drives/Quantum Materials/MagnetTime/2021_07_TLH/'

file_numbers = ['072', '075', '079', '081', '083', '085', '088', '091', '094']

angles = [180, 173, 166, 162.5, 159, 155.5, 152, 145, 138]

tau_root = 'Cant_Y_'

file_root = 'Cambridge_June2021.'

fileandpath_up = main_path + 'Cambridge_June2021.072.txt'

fig, a = MakePlot(figsize=(16, 8), gs=True).create()
gs = fig.add_gridspec(1, 2)

ax1 = fig.add_subplot(gs[:,0])
ax2 = fig.add_subplot(gs[:,1])


for i, num in enumerate(file_numbers):


    field, tau = load_torque(main_path+file_root+str(num)+'.txt', num, tau_root=tau_root,skiprows=7,lower_threshold=8)

    ax1.annotate(str(182 - angles[i]) + r'$\degree$', xy=(0.84, 0.05 + i * 0.95/14), xycoords='axes fraction',
                 fontname='arial', fontsize=24, color=colours[i])
    ax1.plot(field, tau*.12*250e3, linewidth=2, c=colours[i], label=str(angles[i]) + r'$\degree$')
    ax2.plot(field**2, tau*.12*250e3, linewidth=2, c=colours[i], label=str(angles[i])+r'$\degree$')


publication_plot(ax1, r'$\mu_0H$ (T)', r'$\tau$ ($\times 10^{-3}$ $\mu_B$ T per f.u.)')
publication_plot(ax2, r'$(\mu_0H)^2$ (T$^2$)', r'')

ax1.set_xbound(7.5, 50)

plt.tight_layout(pad=1)

plt.savefig(main_path+'raw_data.png', dpi=300, bbox_inches='tight')

plt.show()

