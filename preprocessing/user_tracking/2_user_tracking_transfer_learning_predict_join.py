import pandas as pd
from sklearn.cluster import DBSCAN
import numpy as np
from configurations import USER_TRACKING_LOCAL_DATETIME_OSM_CATEGORIES, USER_TRACKING_LOCAL_DATETIME_GOWALLA_CATEGORIES_TRANSFER_LEARNING_BR, USER_TRACKING_PREDICTED

if __name__ == "__main__":

    df_1 = pd.read_csv(USER_TRACKING_LOCAL_DATETIME_OSM_CATEGORIES).query("country_name == 'Brazil'")

    print("uarios")
    print(len(df_1['id'].unique().tolist()))

    df_2 = pd.read_csv(USER_TRACKING_LOCAL_DATETIME_GOWALLA_CATEGORIES_TRANSFER_LEARNING_BR)

    df_3 = pd.read_csv(USER_TRACKING_LOCAL_DATETIME_GOWALLA_CATEGORIES_TRANSFER_LEARNING_BR)

    df_2 = df_2.join(df_3, on=['user_id', 'poi_id'])

    print(df_2)