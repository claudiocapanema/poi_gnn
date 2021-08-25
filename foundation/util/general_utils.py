import numpy as np
import pandas as pd

def join_df(data:pd.DataFrame, ids:pd.Series, column:str):

    print("tamanho dos ids", len(ids))
    data = data.set_index(column).join(ids, how='inner')
    print("depois: ", data.shape)
    print(data)
    return data