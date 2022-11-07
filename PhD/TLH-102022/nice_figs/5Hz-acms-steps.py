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
file_4 = 'Cambridge_October.028.txt'

file_6 = 'Cambridge_October.056.txt'
file_7 = 'Cambridge_October.033.txt'
file_8 = 'Cambridge_October.049.txt'



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

fig, a = MakePlot(figsize=(6, 8), gs=True).create()
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


dat7 = pd.read_csv(main_path + 'Cambridge_October.033_Up.csv')
field_7 = np.flip(np.abs(dat7['Field_033']))
coil_X_7 = np.flip(dat7['Coil_X_033'])
coil_Y_7 = np.flip(dat7['Coil_Y_033'])
coil_R_7 = np.flip(dat7['Coil_R_033'])

dat8 = pd.read_csv(main_path + 'Cambridge_October.049_Up.csv')
field_8 = np.flip(np.abs(dat8['Field_049']))
coil_X_8 = np.flip(dat8['Coil_X_049'])
coil_Y_8 = np.flip(dat8['Coil_Y_049'])
coil_R_8 = np.flip(dat8['Coil_R_049'])





ax1.plot(field_8, coil_Y_8*1e5, linewidth=.8, c=colours[1], label='$T = 25$ mK')
ax1.plot(field_7, coil_Y_7*1e5, linewidth=.8, c=colours[3], label='$T = 90$ mK')




publication_plot(ax1, r'$\mu_0 H$ (T)', r'$\chi^{\prime\prime}$ (arb.)')#^\prime

ax1.set_ybound(0,18.5)
ax1.set_xbound(0,28)

ax1.annotate(r'$\theta = 30 \degree$', xy=(0.62, 0.15), xycoords='axes fraction',
             ha="left", va="center", fontname='arial', fontsize=24)

ax1.annotate(r'$\omega = 5.5$ Hz', xy=(0.6, 0.07), xycoords='axes fraction',
             ha="left", va="center", fontname='arial', fontsize=24)



legend = ax1.legend(framealpha=0, ncol=1, loc='best',
              prop={'size': 24, 'family': 'arial'}, handlelength=0)

for line,text in zip(legend.get_lines(), legend.get_texts()):
    text.set_color(line.get_color())



plt.tight_layout(pad=1)

plt.savefig(main_path+'temps-steps-5Hz.png', dpi=300, bbox_inches='tight')
# plt.show()



