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

file_name = 'Cambridge_October.008.txt'


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

ax1 = fig.add_subplot(gs[:,0])
ax2 = fig.add_subplot(gs[:,1])


angles, hall = load_xy(main_path+file_name, '008',x_root='Angle_deg_', y_root='Hall_X_')
angles, coil = load_xy(main_path+file_name, '008', x_root='Angle_deg_',y_root='Coil_X_')



ax1.plot(angles, hall, linewidth=2, c=colours[0])

ax2.plot(angles, coil, linewidth=2, c=colours[0])


publication_plot(ax1, r'$\theta (\degree)$', r'$V_{xy}$ (V)')
publication_plot(ax2, r'$\theta (\degree)$', r'$\chi^{\prime}$ (arb.)')



plt.tight_layout(pad=1)
plt.show()



