from pathlib import Path
from matplotlib import pyplot
import numpy as np
import pandas as pd
import os
import time

from loader.file_loader import FileLoader

class MatrixGenerationForPoiCategorizationLoader(FileLoader):

    def __init__(self):
        pass

    def adjacency_features_matrices_to_csv(self,
                                           files, files_names):

        for i in range(len(files)):
            try:
                file = files[i]
                file_name = files_names[i]
                self.save_df_to_csv(file, file_name, 'a')
            except:
                time.sleep(8)
                file = files[i]
                file_name = files_names[i]
                self.save_df_to_csv(file, file_name, 'a')
