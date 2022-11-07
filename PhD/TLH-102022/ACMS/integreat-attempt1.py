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

main_path = '/Volumes/GoogleDrive/Shared drives/Quantum Materials - Sebastian Group/MagnetTime/2022_10_TLH/week1/ACMS/VT154/'


file_1 = 'Cambridge_October.025.txt'
file_2 = 'Cambridge_October.026.txt'
file_3 = 'Cambridge_October.027.txt'


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

fig, a = MakePlot(figsize=(8, 8), gs=True).create()
gs = fig.add_gridspec(1,1)

ax1 = fig.add_subplot(gs[:,0])

# ax2 = fig.add_subplot(gs[:,1])


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

step = 3

field_1, coil_Y_1 = load_xy(main_path+file_1, '025',x_root='Field_', y_root='Coil_Y_',skiprows=38)
field_1, coil_X_1 = load_xy(main_path+file_1, '025', x_root='Field_',y_root='Coil_X_',skiprows=38)
angles_1, coil_R_1 = load_xy(main_path+file_1, '025', x_root='Field_',y_root='Coil_R_',skiprows=38)

field_1 *= -1

interpd_field = np.linspace(0,28,50000)
interpd_X_up = np.interp(interpd_field, field_1, coil_R_1)

locs = np.arange(0,50000,step)

ints = []

avg_fields = []

for l1, l2 in zip(locs, locs[1:]):

    I = np.trapz(interpd_X_up[l1:l2], interpd_field[l1:l2])

    ints.append(I)

    avg_fields.append(np.average(field_1[l1:l2]))


B = np.array(avg_fields)
M = np.array(ints)

ax1.plot(B**2, M*B, linewidth=2, c=colours[0], label='Up 1')

field_2, coil_Y_2 = load_xy(main_path+file_2, '026',x_root='Field_', y_root='Coil_Y_',skiprows=38)
field_2, coil_X_2 = load_xy(main_path+file_2, '026', x_root='Field_',y_root='Coil_X_',skiprows=38)
angles_2, coil_R_2 = load_xy(main_path+file_2, '026', x_root='Field_',y_root='Coil_R_',skiprows=38)

field_2 = -1 * np.flip(field_2)

interpd_field = np.linspace(0,28,50000)
interpd_X_down = np.interp(interpd_field, field_2, coil_R_2)

locs = np.arange(0,50000,step)

ints = []

avg_fields = []

for l1, l2 in zip(locs, locs[1:]):

    I = np.trapz(interpd_X_up[l1:l2], interpd_field[l1:l2])

    ints.append(I)

    avg_fields.append(np.average(field_1[l1:l2]))


B = np.array(avg_fields)
M = np.array(ints)

ax1.plot(B**2, M*B, linewidth=2, c=colours[2], label='Down 1')


field_3, coil_Y_3 = load_xy(main_path+file_3, '027',x_root='Field_', y_root='Coil_Y_',skiprows=38)
field_3, coil_X_3 = load_xy(main_path+file_3, '027', x_root='Field_',y_root='Coil_X_',skiprows=38)
angles_3, coil_R_3 = load_xy(main_path+file_3, '027', x_root='Field_',y_root='Coil_R_',skiprows=38)

field_3 *= -1

interpd_field = np.linspace(0,28,50000)
interpd_X_up = np.interp(interpd_field, field_3, coil_R_3)

locs = np.arange(0,50000,step)

ints = []

avg_fields = []

for l1, l2 in zip(locs, locs[1:]):

    I = np.trapz(interpd_X_up[l1:l2], interpd_field[l1:l2])

    ints.append(I)

    avg_fields.append(np.average(field_1[l1:l2]))


B = np.array(avg_fields)
M = np.array(ints)

ax1.plot(B**2, M*B, linewidth=2, c=colours[0], label='Up 2')




publication_plot(ax1, r'$\mu_0 H$ (T)', r'$M\cdot B$ (arb.)')#^\prime



legend = ax1.legend(framealpha=0, ncol=1, loc='best',
              prop={'size': 24, 'family': 'arial'}, handlelength=0)

for line,text in zip(legend.get_lines(), legend.get_texts()):
    text.set_color(line.get_color())



plt.tight_layout(pad=1)

plt.savefig(main_path+'d1-X-fig.png', dpi=300)
plt.show()



