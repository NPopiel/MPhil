

from tools.utils import *
from tools.ColorMaps import select_discrete_cmap
from scipy.signal import savgol_filter, get_window



file = '/Volumes/GoogleDrive/Shared drives/Quantum Materials/MagnetTime/2022_02_TLH/VT16/day1/-7.5/qos/bgsub_1.5K_up_-7.5deg_5.2-13T_s004.csv'
file2 = '/Volumes/GoogleDrive/Shared drives/Quantum Materials/MagnetTime/2022_02_TLH/VT16/day1/-7.5/qos/bgsub_1.5K_up_-7.5deg_5.2-13T_s006.csv'
file3 = '/Volumes/GoogleDrive/Shared drives/Quantum Materials/MagnetTime/2022_02_TLH/VT16/day1/-7.5/qos/bgsub_inv_1.5K_up_-7.5deg_5.2-13T_s004.csv'
file4 = '/Volumes/GoogleDrive/Shared drives/Quantum Materials/MagnetTime/2022_02_TLH/VT16/day1/-7.5/qos/bgsub_inv_1.5K_up_-7.5deg_5.2-13T_s006.csv'


fig, axs = MakePlot(figsize=(16,8), ncols=1, nrows=2).create()
cmap = select_discrete_cmap('jugglypuff')


field1 = np.genfromtxt(file,delimiter=',')[5:,0]
volts1 = np.genfromtxt(file,delimiter=',')[5:,1]



axs[0].plot(field1, savgol_filter(1e3*volts1, 127, 3), linewidth=2,c=cmap[0])

field2 = np.genfromtxt(file2, delimiter=',')[5:, 0]
volts2 = np.genfromtxt(file2, delimiter=',')[5:, 1]

axs[0].plot(field2, savgol_filter(1e3 * volts2, 127,3), linewidth=2, c=cmap[3])

publication_plot(axs[0], 'Magnetic Field (T)', 'Subtracted Torque (arb.)')

field1 = np.genfromtxt(file3,delimiter=',')[5:,0]
volts1 = np.genfromtxt(file3,delimiter=',')[5:,1]

spacing = (field1[2] - field1[0])/2

fft_vals1 = np.abs(np.fft.rfft(volts1 * get_window('hanning',
                                                      len(field1)),
                              n=20000))

fft_freqs1 = np.fft.rfftfreq(20000, d=spacing)
freq_arg = None
freq_cut = 1200
if freq_cut > 0:
    freq_arg = np.searchsorted(fft_freqs1, freq_cut)
fft1_freqs = fft_freqs1[0:freq_arg]
fft1_vals = fft_vals1[0:freq_arg]

axs[1].plot(fft1_freqs, fft1_vals, linewidth=2, c=cmap[0])

field2 = np.genfromtxt(file4,delimiter=',')[5:,0]
volts2 = np.genfromtxt(file4,delimiter=',')[5:,1]
fft_vals2 = np.abs(np.fft.rfft(volts2 * get_window('hanning',
                                                      len(field2)),
                              n=20000))
fft_freqs2 = np.fft.rfftfreq(20000, d=spacing)
freq_arg = None
freq_cut = 1200
if freq_cut > 0:
    freq_arg = np.searchsorted(fft_freqs2, freq_cut)
fft2_freqs = fft_freqs2[0:freq_arg]
fft2_vals = fft_vals2[0:freq_arg]

axs[1].plot(fft2_freqs, fft2_vals, linewidth=2, c=cmap[3])
publication_plot(axs[1], 'Magnetic Field (T)', 'FFT Amplitude')


legend = axs[1].legend(framealpha=0, ncol=1, loc='best',
              prop={'size': 20, 'family': 'arial'}, handlelength=0)

for line,text in zip(legend.get_lines(), legend.get_texts()):
    text.set_color(line.get_color())


plt.tight_layout(pad=2)
plt.show()