import pandas as pd
from tools.utils import *
from tools.DataFile import DataFile
import seaborn as sns
from tools.MakePlot import *

main_path = '/Volumes/GoogleDrive/Shared drives/Quantum Materials/Popiel/VT49_1p8K_lockin_rotator_1uA/'

filenames = ['VT49_lockin_rotator_deg_0.0.csv']\
''',
             'VT49_lockin_rotator_deg_30.0.csv',
             'VT49_lockin_rotator_deg_45.0.csv',
             'VT49_lockin_rotator_deg_60.0.csv',
             'VT49_lockin_rotator_deg_90.0.csv',
             'VT49_lockin_rotator_deg_120.0.csv',
             'VT49_lockin_rotator_deg_135.0.csv',
             'VT49_lockin_rotator_deg_150.0.csv',
             'VT49_lockin_rotator_deg_180.0.csv']
'''
dat_lst = []
name_lst = []
for filename in filenames:
    dat = pd.read_csv(main_path+filename)
    name_lst.append([filename.split('.')[-3].split('_')[-1]]*len(dat.field))
    dat_lst.append(dat)

df = pd.concat(dat_lst)

df = df.rename(columns={"field": "Magnetic Field (T)", "voltage": "Voltage (V)"})
df['Rotation Angle (degrees)'] = flatten(name_lst)

fig, ax = MakePlot().create()
sns.lineplot(x="Magnetic Field (T)", y="Voltage (V)", hue='Rotation Angle (degrees)', data=df)
plt.title('VT49 Lock-in Voltage by Field and Angle (1.8 K)')
#plt.savefig(main_path+'fig1.png',dpi=600)
#plt.close()
plt.show()



