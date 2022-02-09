import pandas as pd
from sklearn.cluster import DBSCAN
import numpy as np
from configurations import USER_TRACKING_LOCAL_DATETIME_OSM_CATEGORIES, USER_TRACKING_LOCAL_DATETIME_GOWALLA_CATEGORIES_BR

if __name__ == "__main__":

    df = pd.read_csv(USER_TRACKING_LOCAL_DATETIME_OSM_CATEGORIES).query("country_name == 'Brazil'")

    df = df[df['poi_resulting'] != 'Home']
    df = df[df['poi_resulting'] != 'Work']
    df = df[df['poi_resulting'] != 'Commuting']
    df = df[df['poi_resulting'] != 'Other']

    print(df)

    df.to_csv(USER_TRACKING_LOCAL_DATETIME_GOWALLA_CATEGORIES_BR, index=False)