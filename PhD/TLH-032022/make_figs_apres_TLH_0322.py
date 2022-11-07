from tools.utils import *
from tools.ColorMaps import select_discrete_cmap
from scipy.signal import savgol_filter, get_window

file = '/Volumes/GoogleDrive/Shared drives/Quantum Materials/MagnetTime/2022_02_TLH/VT69/good_data/bgsub_0.4K_up_2222deg_25.1-28.2T_s001AGE.dat'

delimiter = '\t'

file2 = '/Volumes/GoogleDrive/Shared drives/Quantum Materials/MagnetTime/2022_02_TLH/VT69/LK/0.4/qos/bgsub_0.4K_up_2222deg_27.2-41.4T_s001.csv'
field_column_number = 0
voltage_column_number = 1 # VT16 5 is signal in Y, 4 is singlan in X, 6 is R

fig, axs = MakePlot(figsize=(16,8), ncols=1, nrows=3).create()
cmap = select_discrete_cmap('jugglypuff')


field1 = np.genfromtxt(file,delimiter=delimiter)[5:,field_column_number]
volts1 = np.genfromtxt(file,delimiter=delimiter)[5:,voltage_column_number]



axs[0].plot(field1, 1e3*volts1, linewidth=2,c=cmap[0])

field2 = np.genfromtxt(file2, delimiter=',')[5:, field_column_number]
volts2 = np.genfromtxt(file2, delimiter=',')[5:, voltage_column_number]

axs[0].plot(field2[field2>31], savgol_filter(1e3 * volts2[field2>31], 127,3), linewidth=2, c=cmap[1])


publication_plot(axs[0], 'Magnetic Field (T)', 'Subtracted Torque (arb.)')

spacing = (field1[2] - field1[0])/2

fft_vals1 = np.abs(np.fft.rfft(volts1 * get_window('hanning',
                                                      len(field1)),
                              n=65536))
fft_freqs1 = np.fft.rfftfreq(65536, d=spacing)
freq_arg = None
freq_cut = 120
if freq_cut > 0:
    freq_arg = np.searchsorted(fft_freqs1, freq_cut)
fft1_freqs = fft_freqs1[0:freq_arg]
fft1_vals = fft_vals1[0:freq_arg]

axs[1].plot(fft1_freqs, fft1_vals, linewidth=2, c=cmap[0])
publication_plot(axs[1], 'Magnetic Field (T)', 'FFT Regime II')

fft_vals2 = np.abs(np.fft.rfft(volts2 * get_window('hanning',
                                                      len(field2)),
                              n=65536))
fft_freqs2 = np.fft.rfftfreq(65536, d=spacing)
freq_arg = None
freq_cut = 120
if freq_cut > 0:
    freq_arg = np.searchsorted(fft_freqs2, freq_cut)
fft2_freqs = fft_freqs2[0:freq_arg]
fft2_vals = fft_vals2[0:freq_arg]

axs[2].plot(fft2_freqs, fft2_vals, linewidth=2, c=cmap[1])
publication_plot(axs[2], 'Magnetic Field (T)', 'FFT Regime III')

# legend = axs[2].legend(framealpha=0, ncol=1, loc='best',
#               prop={'size': 20, 'family': 'arial'}, handlelength=0)
#
# for line,text in zip(legend.get_lines(), legend.get_texts()):
#     text.set_color(line.get_color())


plt.tight_layout(pad=2)
plt.show()