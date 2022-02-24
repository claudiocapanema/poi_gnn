import pandas as pd
from scipy import sparse
import time
import os

class FileLoader:

    def __init__(self):
        pass

    def save_df_to_csv(self, df, filename, mode='w'):
        #filename = DataSources.FILES_DIRECTORY.get_value() + filename
        if os.path.exists(filename):
            header = False
        else:
            header = True
        try:
            df.to_csv(filename, index=False, mode=mode, header=header)
        except:
            time.sleep(8)
            df.to_csv(filename, index=False, mode=mode, header=header)

    def save_sparse_matrix_to_npz(self, matrix, filename):

        try:
            sparse.save_npz(filename, matrix)

        except:
            time.sleep(8)
            sparse.save_npz(filename, matrix)
