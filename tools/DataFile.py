import numpy as np
import pandas as pd
from tools.utils import oersted_to_tesla


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
        new_headers = parameters[3]
        convert_b_flag = parameters[4]
        cols_to_keep = parameters[5]

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

        if new_headers is not None:
            # loop over the columns and update the names
            for ind, new_header in enumerate(new_headers):
                dat[0, ind] = new_header

            # convert to pandas dataframe
            df = pd.DataFrame(dat)

            # Change new labels to the header
            df.columns = df.iloc[0]
            df = df[1:]

            # ensure data is a float, converting all empty and string values to nan
            df = df.apply(pd.to_numeric, errors='coerce')

        else:
            df = pd.DataFrame(dat)
            # Change new labels to the header
            df.columns = df.iloc[0]
            df = df[1:]

            df = df.apply(pd.to_numeric, errors='coerce')

        if convert_b_flag:
            df['b_field'] = oersted_to_tesla(df['b_field'])

        if cols_to_keep is not None:
            df = df[cols_to_keep]

        self.df = df

        return self.df
