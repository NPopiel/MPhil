from tools.utils import *

main_path = '/Users/npopiel/Documents/MPhil/Data/'

file_names = ['FeSb2_data1.csv',
              'FeSb2_data2.csv',
              'SmB6_data1.csv']

for ind, file_name in enumerate(file_names):

    df = pd.read_csv(main_path+'data_csvs_cleaned/'+file_name)

    df, locs=extract_stepwise_peaks(df,'temp','temp_flag','const_temp_')

    groupers = df.groupby('temp_flag')

    new_headers = df.columns

    makedir(main_path+'csvs_by_temp/')
    base_save_path = main_path+'csvs_by_temp/'+file_names[ind].split('.')[0]+'/'
    makedir(base_save_path)

    for constant_temp, inds in groupers.groups.items():
        subsection = df[df.temp_flag == constant_temp]
        temp = round(float(constant_temp.split('_')[-1]))
        save_name = base_save_path+'constant_temp_'+str(temp)+'.csv'
        subsection.to_csv(save_name,index=False)

