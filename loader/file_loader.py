import pandas as pd
from scipy import sparse
import time

class FileLoader:

    def __init__(self):
        pass

    def save_df_to_csv(self, df, filename):
        #filename = DataSources.FILES_DIRECTORY.get_value() + filename
        try:
            df.to_csv(filename, index=False)
        except:
            time.sleep(8)
            df.to_csv(filename, index=False)

    def save_sparse_matrix_to_npz(self, matrix, filename):

        try:
            sparse.save_npz(filename, matrix)

        except:
            time.sleep(8)
            sparse.save_npz(filename, matrix)
