import pandas as pd
from tools.utils import *
from tools.DataFile import DataFile
import seaborn as sns
from tools.MakePlot import *

main_path = '/Volumes/GoogleDrive/Shared drives/Quantum Materials/Popiel/VT49_1p8K_lockin_rotator_1uA/'

filenames = ['VT49_lockin_rotator_deg_0.0.csv',
             'VT49_lockin_rotator_deg_30.0.csv',
             'VT49_lockin_rotator_deg_45.0.csv',
             'VT49_lockin_rotator_deg_60.0.csv',
             'VT49_lockin_rotator_deg_90.0.csv',
             'VT49_lockin_rotator_deg_120.0.csv',
             'VT49_lockin_rotator_deg_135.0.csv',
             'VT49_lockin_rotator_deg_150.0.csv',
             'VT49_lockin_rotator_deg_180.0.csv']

def sweeps_split_easy(dat):
    len_arr = len(dat["Magnetic Field (T)"])
    mid_loc = len_arr / 2 if len_arr % 2 == 0 else (len_arr - 1) / 2
    max_loc = np.argmax(dat["Magnetic Field (T)"])
    min_loc = np.argmin(dat["Magnetic Field (T)"])
    lst = []
    for val in np.arange(len_arr):
        if val <=max_loc:
            lst.append('0->14T')
        elif max_loc < val <= min_loc:
            lst.append('14T->-14T')
        else:
            lst.append('-14T->0T')
    return lst


dat_lst = []

for filename in filenames:
    dat = pd.read_csv(main_path+filename)
    dat.voltage = 1000000 * np.array(dat.voltage)
    dat = dat.rename(columns={"field": "Magnetic Field (T)", "voltage": r"Resistance $(\Omega)$"})
    dat['Field Sweep'] = sweeps_split_easy(dat)
    dat['Rotation Angle (degrees)'] = [filename.split('.')[-3].split('_')[-1]]*len(dat["Magnetic Field (T)"])
    dat_lst.append(dat)

df = pd.concat(dat_lst)

angles = [0,30,45,60,90,120,135,150,180]
fig, ax = MakePlot().create()

g = sns.FacetGrid(df, col="Rotation Angle (degrees)", col_wrap=3, hue='Field Sweep')
g.map(sns.lineplot, "Magnetic Field (T)", r"Resistance $(\Omega)$", lw=0.45)#, style='Field Sweep')
#g.add_legend()
plt.legend()
plt.suptitle('VT49 Lock-in Voltage by Field Sweep and Angle (1.8 K)')
plt.tight_layout()
#plt.savefig(main_path+'fig2.png',dpi=600)
#plt.close()
plt.show()

'''

groupers = df.groupby('Rotation Angle (degrees)')
fig, ax = MakePlot(nrows=3,ncols=3).create()
c=0
axs = [ax[0,0],ax[0,1],ax[0,2],ax[1,0],ax[1,1],ax[1,2],ax[2,0],ax[2,1],ax[2,2]]
angles = [0,30,45,60,90,120,135,150,180]
for angle, idx in groupers.groups.items():
    df_pos = df[df['Rotation Angle (degrees)'] == angle]
    sns.scatterplot(x="Magnetic Field (T)", y=r"Resistance $(\Omega)$", data=df_pos, style='Field Sweep', hue='Field Sweep',ax=axs[c])
    axis = plt.gca()
    plt.title(str(angles[c])+'Degrees')
    if c != 0 or c != 3 or c !=6:
        axis.set_ylabel('')
    if c != 6 or c != 7 or c !=8:
        axis.set_xlabel('')
    c+=1

plt.legend()
plt.suptitle('VT49 Lock-in Voltage by Field Sweep and Angle (1.8 K)')
plt.tight_layout()
#plt.savefig(main_path+'fig2.png',dpi=600)
#plt.close()
plt.show()
'''

