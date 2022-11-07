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

file_up = 'Cambridge_October.016.txt'
file_dn = 'Cambridge_October.017.txt'
file_dndn = 'Cambridge_October.018.txt'


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
gs = fig.add_gridspec(1,3)

ax1 = fig.add_subplot(gs[:,0])
ax2 = fig.add_subplot(gs[:,1])
ax3 = fig.add_subplot(gs[:,2])
# ax2 = fig.add_subplot(gs[:,1])


field_up, coil_Y_up = load_xy(main_path+file_up, '016',x_root='Field_', y_root='Coil_Y_',skiprows=14, space_x = True)
field_up, coil_X_up = load_xy(main_path+file_up, '016', x_root='Field_',y_root='Coil_X_',skiprows=14, space_x = True)
field_up, coil_R_up = load_xy(main_path+file_up, '016', x_root='Field_',y_root='Coil_R_',skiprows=14, space_x=True)

field_dn, coil_Y_dn = load_xy(main_path+file_dn, '017',x_root='Field_', y_root='Coil_Y_',skiprows=13)
field_dn, coil_X_dn = load_xy(main_path+file_dn, '017', x_root='Field_',y_root='Coil_X_',skiprows=13)
angles_dn, coil_R_dn = load_xy(main_path+file_dn, '017', x_root='Field_',y_root='Coil_R_',skiprows=13)

field_dndn, coil_Y_dndn = load_xy(main_path+file_dndn, '018',x_root='Field_', y_root='Coil_Y_',skiprows=13)
field_dndn, coil_X_dndn = load_xy(main_path+file_dndn, '018', x_root='Field_',y_root='Coil_X_',skiprows=13)
angles_dndn, coil_R_dndn = load_xy(main_path+file_dndn, '018', x_root='Field_',y_root='Coil_R_',skiprows=13)



ax1.plot(field_up, coil_X_up, linewidth=2, c='#048BA8')
ax1.plot(field_dn, coil_X_dn, linewidth=2, c='#734B5E')
ax1.plot(field_dndn, coil_X_dndn, linewidth=2, c='#C7EBF0')

ax2.plot(field_up, coil_Y_up, linewidth=2, c='#048BA8')
ax2.plot(field_dn, coil_Y_dn, linewidth=2, c='#734B5E')
ax2.plot(field_dndn, coil_Y_dndn, linewidth=2, c='#C7EBF0')

ax3.plot(field_up, coil_R_up, linewidth=2, c='#048BA8')
ax3.plot(field_dn, coil_R_dn, linewidth=2, c='#734B5E')
ax3.plot(field_dndn, coil_R_dndn, linewidth=2, c='#C7EBF0')


publication_plot(ax1, r'$\mu_0 H$ (T)', r'$\chi^\prime$ (arb.)')
publication_plot(ax2, r'$\mu_0 H$ (T)', r'$\chi^{\prime\prime}$ (arb.)')
publication_plot(ax3, r'$\mu_0 H$ (T)', r'$\chi$ (arb.)')


legend = ax1.legend(framealpha=0, ncol=1, loc='best',
              prop={'size': 24, 'family': 'arial'}, handlelength=0)

for line,text in zip(legend.get_lines(), legend.get_texts()):
    text.set_color(line.get_color())



plt.tight_layout(pad=1)
plt.show()



