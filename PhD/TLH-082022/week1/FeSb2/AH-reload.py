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

resistance_path = '/Volumes/GoogleDrive/Shared drives/Quantum Materials/MagnetTime/2022_08_TLH/week1/FeSb2-VT54/resistance/AH/'
torque_path = '/Volumes/GoogleDrive/Shared drives/Quantum Materials/MagnetTime/2022_08_TLH/week1/FeSb2-VT54/torque/AH/'

fig, a = MakePlot(figsize=(12, 12), gs=True).create()
gs = fig.add_gridspec(1, 1)

ax1 = fig.add_subplot(gs[0,:3])

# Draw image
axin = ax1.inset_axes([0.1,.2,.4,.4])    # create new inset axes in data coordinates


sweeps = ['0.29K_103.33deg_sweep093_down.csv']
lstyles = ['solid', 'dashed']

labels = ['Up', 'Down']

for i, s_name in enumerate(sweeps):


    torque_dat = load_matrix(torque_path + s_name)
    field = torque_dat[:, 0]
    tau = torque_dat[:, 1]


    resistance_dat = load_matrix(resistance_path + s_name)
    B = resistance_dat[:, 0]
    V = resistance_dat[:, 1]  # mV

    I = 10e-6 #A

    R = V / I #Ohm

    ax1.plot(field, tau, linewidth=2, c='#0A9396')
    axin.plot(B, R, linewidth=2, c='#9B2226')

axin.set_ybound(0, 500)

publication_plot(ax1, r'$\mu_0H$ (T)', r'$\tau$ (pF)')
publication_plot(axin, r'$\mu_0H$ (T)', r'$R$ ($\Omega$)',label_fontsize=18, tick_fontsize=16)

plt.show()




