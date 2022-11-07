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

file_name = 'Cambridge_October.014.txt'


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

fig, a = MakePlot(figsize=(16, 8), gs=True).create()
gs = fig.add_gridspec(2,2)

ax1 = fig.add_subplot(gs[:,:])
# ax2 = fig.add_subplot(gs[:,1])


angles, coil_Y = load_xy(main_path+file_name, '014',x_root='Angle_deg_', y_root='Coil_Y_')
angles, coil_X = load_xy(main_path+file_name, '014', x_root='Angle_deg_',y_root='Coil_X_')
angles, coil_R = load_xy(main_path+file_name, '014', x_root='Angle_deg_',y_root='Coil_R_')

angles -= 8

ax1.plot(angles, coil_X, linewidth=2, c=colours[1], label=r'$\chi^{\prime}$')
ax1.plot(angles, coil_Y, linewidth=2, c=colours[3], label=r'$\chi^{\prime\prime}$')


publication_plot(ax1, r'$\theta (\degree)$', r'$\chi$ (arb.)')

ax1.set_ybound(-0.0068, 0.0058)
ax1.set_xbound(-39, 36)

legend = ax1.legend(framealpha=0, ncol=1, loc='best',
              prop={'size': 24, 'family': 'arial'}, handlelength=0)

for line,text in zip(legend.get_lines(), legend.get_texts()):
    text.set_color(line.get_color())



plt.tight_layout(pad=1)
plt.show()



