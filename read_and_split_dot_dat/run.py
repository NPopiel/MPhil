import seaborn as sns
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import scipy.io
from scipy.signal import find_peaks

class MakePlot():
     # This class is used to initalise figures and ensure they all look the same
    def __init__(self, nrows=1,ncols=1,figsize=(12,9)):
        self.nrows = nrows
        self.ncols = ncols
        self.figsize = figsize

    def create(self):
        matplotlib.use('TkAgg')
        plt.rc('font', family='serif', size=14)
        sns.set_context('paper')
        fig, axs = plt.subplots(nrows=self.nrows, ncols=self.ncols,figsize=self.figsize)
        plt.interactive(False)
        return fig, axs

class DataFile:
    # Initializer / Instance Attributes
    def __init__(self, filename, parameters):
        self.filename = filename
        self.parameters = parameters
        self.df = pd.DataFrame()

    # method for opening the file. Fairly straightforward.

    def open(self):

        flatten = lambda l: [item for sublist in l for item in sublist]

        parameters = self.parameters

        delimeter = parameters[0]
        row_after_header_useless = parameters[1]
        delete_comment_flag = parameters[2]


        file = open(self.filename,'r')

        num_to_skip = 0

        for ind, line in enumerate(file.readlines()):
            if line == '[Data]\n':
                num_to_skip = ind + 1
                break

        file.close()

        with open(self.filename, 'r') as the_file:
            all_data = [line.split(delimeter) for line in the_file.readlines()[num_to_skip:]]
            if row_after_header_useless: all_data.pop(1)
            dat_arr = np.array(flatten(all_data), dtype='U64').reshape((len(all_data), len(all_data[0])))

        if delete_comment_flag:
            dat = np.delete(dat_arr, 0, 1)
            del dat_arr
        else:
            dat = dat_arr
            del dat_arr

        df = pd.DataFrame(dat)
        # Change new labels to the header
        df.columns = df.iloc[0]
        df = df[1:]

        df = df.apply(pd.to_numeric, errors='coerce')

        self.df = df

        return self.df

flatten = lambda l: [item for sublist in l for item in sublist]

def extract_stepwise_temps(df, col_name, new_col_name,  plot=False,threshold=1, round_num=1, last_point=True):

    # looks at first difference in temperature, finds where larger than threshhold and labels accordingly
    # can plot to verify if correct

    relevant_parameter = np.abs(np.array(df[col_name]))
    time_deriv = np.diff(relevant_parameter)
    peaks = np.where(time_deriv>=threshold)[0]
    peaks+=1

    if last_point: locs = flatten([[0],peaks.tolist(),[len(relevant_parameter)]])
    else: locs = flatten([[0],peaks.tolist()[:-1], [len(relevant_parameter)]])

    lst = []


    new_arr = np.zeros(len(relevant_parameter),dtype='S32')

    for idx_pair in zip(locs[:-1],locs[1:]):
        for j in range(idx_pair[0],idx_pair[1]):
            indexes = np.arange(idx_pair[0],idx_pair[1])

            t = str(np.mean(relevant_parameter[indexes]).round(round_num))
            temp_string = ''
            for char in t:
                res = char
                if char == '.':
                    res = 'p'
                temp_string += res
            lst.append(temp_string+'K')


    df[new_col_name] = lst

    if plot:
        fig, ax = MakePlot().create()
        plt.plot(relevant_parameter)
        for loc in locs: plt.axvline(loc)
        plt.show()

    return df, locs

def extract_current_peaks(df, col_name, new_col_name, distance_between_peaks=50, plot=False):

    # extracts the current peaks and labels the regions of constant current accordingly.
    # can plot if we want to verify

    relevant_parameter = np.array(df[col_name])
    time_deriv = np.diff(relevant_parameter)

    peaks, properties = find_peaks(time_deriv, distance=distance_between_peaks)
    if plot:
        fig, ax = MakePlot().create()
        plt.plot(time_deriv)
        plt.show()
    peaks+=1
    locs = flatten([[0],peaks.tolist(),[len(relevant_parameter)]])
    lst = []


    for idx_pair in zip(locs[:-1],locs[1:]):

        curr = int(round(relevant_parameter[idx_pair[0]+1],0))
        num = idx_pair[1]-idx_pair[0]
        lst.append(num * [str(curr)+'uA'])



    df[new_col_name] = flatten(lst)

    return df, locs

def load_matrix(filepath, delimeter=',',params=[',',True,True]):

    # loads in many different types of files. If you want a .txt file, specify the delimeter

    extension = filepath.split('.')[-1]
    if str(extension) == 'csv':
        return np.genfromtxt(filepath,delimiter = delimeter)
    elif str(extension) == 'txt':
        return np.genfromtxt(filepath,delimiter = delimeter, names=True)
    elif str(extension) == 'npy':
        return np.load(filepath)
    elif str(extension) == 'mat':
        return scipy.io.loadmat(filepath)
    elif str(extension) == 'npz':
        return np.load(filepath)
    elif str(extension) == 'dat':
        return DataFile(filepath,parameters=params).open()

def file_exists(filename):

    # checks if file at the given filename exists or not. Returns a boolean variable

    exists = os.path.isfile(filename)
    if exists:
        return True
    else:
        return False

def save_file(data,path,name,file_check=True):

    # saves files as either csv if below dimension two, or numpy npy files if not. Looking to expand functinality
    # for other types of files.

    if hasattr(data, '__len__') and (not isinstance(data, str)):
        data = np.asarray(data)


    default_delimiter = ','
    format = '%1.5f'

    if len(data.shape) <= 2:
        file = path + str(name) + '.csv'
        if file_check:
            if not file_exists(file):
                np.savetxt(file, data, delimiter=default_delimiter, fmt=format)
        else:
            np.savetxt(file, data, delimiter=default_delimiter, fmt=format)
    else:
        file = path + str(name) + '.npy'
        if file_check:
            if not file_exists(file):
                np.save(file,[data])
        else:
            np.save(file, [data])


# Put in path to Data
main_path = '/Users/npopiel/Documents/MPhil/Data/step_data/'

# the path where we want to save the data. Same as main oath for me.
save_path = main_path

# filename with any extra folder
# Multiple filenames can be put together in a list, but since the file for NYE broke, I did not want to corrupt the data
filenames = [#'VT64_stepmapping.dat']#,
             'VT64_stepmapping_NYEedition600700uA.dat']

# Note that NYEEdition gets wrecked at 1242

# name which will be used to save the file
sample_name_save = 'VT64_NYE'

# relevant columns needed, allows us to clear memory
relevant_columns = ['Temperature (K)',
                    'Magnetic Field (Oe)',
                    'Bridge 2 Resistance (Ohms)',
                    'Bridge 2 Excitation (uA)']


for ind, name in enumerate(filenames):
    # get full filename and path, load in the file and only keep relevant columns
    filename = main_path + name

    df = load_matrix(filename)

    df = df[relevant_columns]

    # This line drops the shitty parts of the file, only necessary if file gets broken down. Only necessary for NYE

    df = df.drop(np.arange(1242, 1293))

    # threshold is the difference in the steps! since we go in increments of 0.25 I say keep bigger than 0.24
    # if you want to verify it is done correctly un comment the line with plot=True

    df, locs = extract_stepwise_temps(df, 'Temperature (K)', 'temp', threshold=0.24, round_num=2)
    #df, locs = extract_stepwise_temps(df, 'Temperature (K)', 'temp', threshold=0.24, plot=True, round_num=2)
    df = df.reset_index()

    # groupe data by temp to loop over and extract current changes
    groupers = df.groupby('temp')

    for constant_temp, inds in groupers.groups.items():

        df_T = df[df.temp == constant_temp]

        # if you want to verify it is done correctly uncomment the line with plot=True

        #df_T, peaks_current = extract_current_peaks(df_T, 'Bridge 2 Excitation (uA)', 'current', plot=True)
        df_T, peaks_current = extract_current_peaks(df_T, 'Bridge 2 Excitation (uA)', 'current')

        # group by current to further loop over
        groupers2 = df_T.groupby('current')

        for current, indxs in groupers2.groups.items():

            # select data only with constant current, and constant temperature

            subsection = df_T[df_T['current'] == current]

            # get resistance and field values (after converting to tesla)

            resistance = subsection['Bridge 2 Resistance (Ohms)']
            field = subsection['Magnetic Field (Oe)']/10000

            # filename for saving, converting to an array with resistance in first col, field in second then saving!

            filename = sample_name_save + constant_temp + current

            array = np.array([resistance, field]).T

            save_file(array, save_path, filename, file_check=False)
