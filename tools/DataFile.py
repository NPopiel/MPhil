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

        num_to_skip = parameters[0]
        delimeter = parameters[1]
        row_after_header_useless = parameters[2]
        delete_comment_flag = parameters[3]
        new_headers = parameters[4]
        convert_b_flag = parameters[5]
        cols_to_remove = parameters[6]

        # FIgure out how to skip the header

        with open(self.filename, 'r') as the_file:
            all_data = [line.split(delimeter) for line in the_file.readlines()[num_to_skip:]]
            if row_after_header_useless: all_data.pop(1)
            dat_arr = np.array(flatten(all_data)).reshape((len(all_data), len(all_data[0])))


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
            df = df.apply(pd.to_numeric, errors='coerce')

        if convert_b_flag:
            df['b_field'] = oersted_to_tesla(df['b_field'])

        if cols_to_remove is not None:
            df.drop(cols_to_remove)

        self.df = df

        return self.df
