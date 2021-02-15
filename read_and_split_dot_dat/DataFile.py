import numpy as np
import pandas as pd
from tools.constants import *

def oersted_to_tesla(arr):
    return arr/OERSTED_2_TESLA

class DataFile:
    # Initializer / Instance Attributes
    def __init__(self, filename, parameters):
        self.filename = filename
        self.parameters = parameters
        self.df = pd.DataFrame()



    def open(self):

        flatten = lambda l: [item for sublist in l for item in sublist]

        parameters = self.parameters

        delimeter = parameters[0]
        row_after_header_useless = parameters[1]
        delete_comment_flag = parameters[2]

        # FIgure out how to skip the header

        file = open(self.filename,'r')

        num_to_skip = 0

        for ind, line in enumerate(file.readlines()):
            if line == '[Data]\n':
                num_to_skip = ind + 1
                break

        file.close()
        #all_data = []
        #file = open(self.filename,'r')
        # for line in file.readlines()[num_to_skip:]:
        #     all_data.append(line.split(delimeter))

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
