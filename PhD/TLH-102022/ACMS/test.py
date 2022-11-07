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


def load_xy(fileandpath, filenumber, up=True,y_root='Coil_X_', x_root = 'Field_'):
    dat  = pd.read_csv(fileandpath, delimiter='\t', skiprows=9)

    x = np.array(dat[x_root + str(filenumber)])
    y = np.array(dat[y_root + str(filenumber)])

    if not up:
        x = np.flip(x)
        y = np.flip(y)
    return x, y

main_path = '/Volumes/GoogleDrive/Shared drives/Quantum Materials - Sebastian Group/MagnetTime/2022_10_TLH/week1/ACMS/VT154/'

file_up = 'Cambridge_October.004.txt'
file_down = 'Cambridge_October.005.txt'

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

fig, a = MakePlot(figsize=(10, 12), gs=True).create()
gs = fig.add_gridspec(3,2)

ax1 = fig.add_subplot(gs[:2,0])
ax2 = fig.add_subplot(gs[:2,1])
ax3 = fig.add_subplot(gs[2,0])
ax4 = fig.add_subplot(gs[2,1])

field_up, up_x = load_xy(main_path+file_up, '004')
field_up, up_y = load_xy(main_path+file_up, '004', y_root='Coil_Y_')

up_x /= 1e-3
up_y /= 1e-4

field_dn, dn_x = load_xy(main_path+file_down, '005',up=False)
field_dn, dn_y = load_xy(main_path+file_down, '005', y_root='Coil_Y_', up=False)

dn_x /= 1e-3
dn_y /= 1e-4

ax1.plot(field_up, up_x, linewidth=2, c=colours[0], label='Up')
ax1.plot(field_dn, dn_x, linewidth=2, c=colours[3], label='Down')

interpd_field = np.linspace(0,9,10000)
interpd_X_up = np.interp(interpd_field, field_up, up_x)
interpd_X_dn = np.interp(interpd_field, field_dn, dn_x)
interpd_Y_up = np.interp(interpd_field, field_up, up_y)
interpd_Y_dn = np.interp(interpd_field, field_dn, dn_y)

ax3.plot(interpd_field, interpd_X_up - interpd_X_dn, linewidth=2, c='darkslategray')
ax4.plot(interpd_field, interpd_Y_up - interpd_Y_dn, linewidth=2, c='darkslategray')

ax2.plot(field_up, up_y, linewidth=2, c=colours[0])
ax2.plot(field_dn, dn_y, linewidth=2, c=colours[3])

publication_plot(ax1, r'$\mu_0 H$ (T)', r'$\chi^\prime$ (arb.)')
publication_plot(ax2, r'$\mu_0 H$ (T)', r'$\chi^{\prime\prime}$ (arb.)')

publication_plot(ax3, r'$\mu_0 H$ (T)', r'$\Delta\chi^\prime$ (arb.)')
publication_plot(ax4, r'$\mu_0 H$ (T)', r'$\Delta\chi^{\prime\prime}$ (arb.)')

legend = ax1.legend(framealpha=0, ncol=1, loc='best',
              prop={'size': 24, 'family': 'arial'}, handlelength=0)

for line,text in zip(legend.get_lines(), legend.get_texts()):
    text.set_color(line.get_color())

plt.tight_layout(pad=1)
plt.show()



