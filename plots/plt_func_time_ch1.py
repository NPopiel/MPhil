import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import pandas as pd
from tools.utils import *
from scipy.signal import find_peaks
from tools.MakePlot import MakePlot
from matplotlib.offsetbox import AnchoredText

main_path = '/Users/npopiel/Documents/MPhil/Data/csvs_by_temp/'

folder_exts = ['FeSb2_data1',
              'FeSb2_data2']
              #,'SmB6_data1']

base_file_name = 'constant_temp_'
file_ext = '.csv'

temp_ranges = [(2,30),
               (2,10)]
               #,(2,23)]

headers_ch1 = ['b_field', 'chamber_pressure', 'sigma_resistance_ch1',
               'phase_angle_ch1', 'ac_current_ch1', 'voltage_amp_ch1', 'in_phase_voltage_ch1',
               'quad_voltage_ch1', 'gain_ch1', 'second_harmonic_ch1', 'third_harmonic_ch1']
headers_ch2 = ['b_field', 'chamber_pressure', 'sigma_resistance_ch2', 'phase_angle_ch2', 'ac_current_ch2',
               'voltage_amp_ch2',
               'in_phase_voltage_ch2', 'quad_voltage_ch2', 'gain_ch2', 'second_harmonic_ch2',
               'third_harmonic_ch2']

current_names = ['I = 1.0', 'I = 5.0', 'I = 10.0', 'I = 20.0','I = 50.0','I = 100.0' , 'I = 200.0' ,  'I = 500.0','I = 1000.0', 'I = 1500.0' ]

for idx, folder in enumerate(folder_exts):
    open_path = main_path+folder+'/'
    t1 = temp_ranges[idx][0]
    t2 = temp_ranges[idx][1]+1
    save_path = open_path + 'figures/'
    makedir(save_path)
    save_path = save_path + 'ch1/'
    makedir(save_path)
    for temp in range(t1,t2):
        file_name = open_path+base_file_name+str(temp)+file_ext
        df = pd.read_csv(file_name)

        df, peaks_voltage = extract_sweep_peaks(df,'voltage_amp_ch1','voltage_sweep_ch1','sweep_')

        df, peaks_current = extract_sweep_peaks(df, 'ac_current_ch1','current_sweep_ch1','I = ')

        fig, axs = MakePlot(ncols=3,nrows=3,figsize=(16,9)).create()
        ax1 = axs[0, 0]
        ax2 = axs[0, 1]
        ax3 = axs[0, 2]
        ax4 = axs[1, 0]
        ax5 = axs[1, 1]
        ax6 = axs[1, 2]
        ax7 = axs[2, 0]
        ax8 = axs[2, 1]
        ax9 = axs[2, 2]

        sns.scatterplot(y='resistance_ch1', x='time',hue='current_sweep_ch1', data=df, ax=ax1,legend=False)
        #ax1.set_title('Resistance by Time')
        ax1.set_ylabel(r'$R  $', usetex=True, rotation=0)
        ax1.set_xlabel('')

        sns.scatterplot(y='ac_current_ch1', x='time', hue='current_sweep_ch1', data=df, ax=ax2,legend=True)
        #ax2.set_title('Resistance by Time')
        ax2.set_ylabel(r'$I  $', usetex=True, rotation=0)
        ax2.set_xlabel('')

        sns.scatterplot(y='voltage_amp_ch1', x='time',hue='current_sweep_ch1', data=df, ax=ax3,legend=False)
        #ax3.set_title('Voltage by Time')
        ax3.set_ylabel(r'$V  $', usetex=True, rotation=0)
        ax3.set_xlabel('')

        sns.scatterplot(y='phase_angle_ch1', x='time', hue='current_sweep_ch1', data=df, ax=ax4,legend=False)
        #ax4.set_title('Resistance by Time')
        ax4.set_ylabel(r'$\phi  $', usetex=True, rotation=0)
        ax4.set_xlabel('')

        sns.scatterplot(y='b_field', x='time', hue='current_sweep_ch1', data=df, ax=ax5,legend=False)
        #ax5.set_title('Magnetic Field by Time')
        ax5.set_ylabel(r'$B  $', usetex=True, rotation=0)
        ax5.set_xlabel('')

        sns.scatterplot(y='sigma_resistance_ch1', x='time', hue='current_sweep_ch1', data=df, ax=ax6,legend=False)
        #ax6.set_title(r'$\sigma_R$  by Time', usetex=True)
        ax6.set_ylabel(r'$\sigma_R$', usetex=True, rotation=0)
        ax6.set_xlabel('')

        sns.scatterplot(y='in_phase_voltage_ch1', x='time', hue='current_sweep_ch1', data=df, ax=ax7,legend=False)
        #ax7.set_title('In Phase Voltage by Time')
        ax7.set_ylabel(r'$V_{\phi}  $', usetex=True, rotation=0)
        ax7.set_xlabel(r'$t (s)$', usetex=True)

        sns.scatterplot(y='quad_voltage_ch1', x='time', hue='current_sweep_ch1', data=df, ax=ax8,legend=False)
        #ax8.set_title(r'Quadrature Voltage  by Time', usetex=True)
        ax8.set_ylabel(r'$V_{Q}  $', usetex=True, rotation=0)
        ax8.set_xlabel(r'$t (s)$', usetex=True)

        sns.scatterplot(y='gain_ch1', x='time', hue='current_sweep_ch1', data=df, ax=ax9,legend=False)
        #ax9.set_title(r'Gain  by Time', usetex=True)
        ax9.set_ylabel(r'Gain ', usetex=True, rotation=0)
        ax9.set_xlabel(r'$t (s)$', usetex=True)

        legend = ax2.legend()
        ax2.get_legend().remove()

        fig.suptitle('Electronic Properties in Time (T = '+str(temp)+' K)',fontsize=22)
        plt.figlegend(title='Current (mA)',frameon=False,loc='center right')#,labels=np.unique(df.current_sweep_ch2.values), frameon=True)
        name = save_path+'change_in_time_t_'+str(temp)+'.pdf'
        plt.savefig(name,dpi=200)
        #plt.show()
        print('Done!', name)
        plt.close()

        '''
        sns.scatterplot(y='resistance_ch2', x='voltage_amp_ch2',hue='current_sweep_ch2', data=df, ax=ax1)
        ax1.set_title('Resistance by Voltage')
        axs.set_ylabel(r'Resistance $(\Omega)$', usetex=True, rotation=0)
        axs.set_xlabel(r'Voltage $(V)$', usetex=True)
        #plt.legend(title="Current (mA)", frameon=True)

        sns.scatterplot(y='resistance_ch2', x='ac_current_ch2',hue='current_sweep_ch2', data=df, ax=ax2)
        ax1.set_title('Resistance by Current')
        axs.set_ylabel(r'R (\Omega)', usetex=True, rotation=0)
        axs.set_xlabel(r'Current $(A)$', usetex=True)
        legend = plt.legend(title="Current (mA)", frameon=True)





        plt.show()
        print('')
        
        '''
