import pandas as pd
from tools.utils import *
from tools.DataFile import DataFile
import seaborn as sns

main_path = '/Volumes/GoogleDrive/Shared drives/Quantum Materials/Popiel/'

filename = 'VT49&VT1_rotator003.txt'

samples = ['VT49','VT1']

file = load_matrix(main_path+filename, delimeter='\t')

df = pd.DataFrame(file)

names = df.columns

names_to_keep = ['PPMS_Field_T','PPMS_Temp_K','PPMS_Position_deg','SR865A2_X_V']

renamed = ['field', 'temp','angle','voltage']

df = df[names_to_keep]

df['PPMS_Position_deg'] = np.abs(df['PPMS_Position_deg'])

df, locs = extract_stepwise_peaks(df, 'PPMS_Position_deg', 'pos_switch','deg_',)

groupers = df.groupby('pos_switch')

for pos, idx in groupers.groups.items():

    if len(idx)>10:

        save_name = 'VT49_lockin_rotator_' + pos + '.csv'

        df_pos = df[df.pos_switch == pos]

        for ind, name in enumerate(names_to_keep): df_pos[renamed[ind]] = df_pos[name]

        df_pos = df_pos[renamed].reset_index()
        inds = []
        for ind, field in enumerate(df_pos.field):
            if np.abs(field)< 0.15:
                inds.append(ind)

        df_pos = df_pos.drop(inds)

        df_pos = df_pos[['field', 'voltage']]

        df_pos.to_csv(main_path+save_name,index=False)

    else:
        print('code bug')


