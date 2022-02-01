import pandas as pd
from sklearn.cluster import DBSCAN
import numpy as np
from configurations import USER_TRACKING_LOCAL_DATETIME_OSM_CATEGORIES


def cluster_geo_data(points, eps, min_samples):
    p_radians = np.radians(points)
    db = DBSCAN(eps=eps, min_samples=min_samples, algorithm='ball_tree', metric='haversine').fit(p_radians)
    labels = db.labels_
    return db, labels

def add_user_pois_ids(user, poi_id_count):

    latitude = user['latitude'].tolist()
    longitude = user['longitude'].tolist()
    coordinates = np.asarray([(latitude[i], longitude[i]) for i in range(len(latitude))], dtype=np.float64)
    eps = 0.03 / 6371.0088
    min_samples = 8
    db, labels = cluster_geo_data(coordinates, eps, min_samples)
    poi_resulting = user['poi_resulting'].tolist()
    poi_id = []

    poi_id_count += 1
    poi_id_commuting = poi_id_count
    poi_id_dict = {-1: poi_id_commuting}
    for i in range(len(labels)):
        if labels[i] != -1:
            poi_id_count += 1
            poi_id_dict[labels[i]] = poi_id_count

    for i in range(len(labels)):

        poi_id.append(poi_id_dict[labels[i]])

    # verification
    unique_pois_ids = pd.Series(poi_id).unique().tolist()
    for i in range(1, len(poi_id)):

        previous_poi_id = poi_id[i-1]
        current_poi_id = poi_id[i]
        previous_resulting_category = poi_resulting[i-1]
        current_poi_resulting = poi_resulting[i]

        if previous_poi_id == current_poi_id and previous_resulting_category != current_poi_resulting:

            print("erro")
            print(user[['latitude', 'longitude', 'poi_resulting', 'poi_type', 'poi_osm']].iloc[i-1])
            print(user[['latitude', 'longitude', 'poi_resulting', 'poi_type', 'poi_osm']].iloc[i])

            exit()

    columns = user.columns.tolist()
    select = []
    for i in range(len(columns)):

        if columns[i] != 'id':

            select.append(columns[i])

    print("colunas", select)
    user['poi_id'] = np.array(labels)
    user = user[columns + ['poi_id']]

    return user


if __name__ == "__main__":

    df = pd.read_csv(USER_TRACKING_LOCAL_DATETIME_OSM_CATEGORIES)

    poi_id_count = 0
    df = df.groupby('id').apply(lambda e: add_user_pois_ids(e, poi_id_count))
    print(df)
