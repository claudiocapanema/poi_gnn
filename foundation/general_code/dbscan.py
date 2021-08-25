import numpy as np
from sklearn.cluster import DBSCAN

class Dbscan:

    def __init__(self, points, min_samples, eps):
        self.points = points
        self.min_samples = min_samples
        self.eps = eps

    def cluster_geo_data(self):
        p_radians = np.radians(self.points)
        db = DBSCAN(eps=self.eps, min_samples=self.min_samples, algorithm='ball_tree', metric='haversine').fit(p_radians)
        self.db = db
        return db

    def get_clusters_with_points_datatime_and_durations(self, datetime_list: list, durations_list: list):
        pois_coordinates = {}
        labels = self.db.labels_
        pois_times = {}
        pois_durations = {}

        for i in range(len(list(set(labels)))):
            if i != -1:
                pois_coordinates[i] = []
                pois_times[i] = []
                pois_durations[i] = []
        size = min([len(self.points), len(datetime_list)])
        for i in range(size):
            if labels[i] != -1:
                pois_coordinates[labels[i]].append(self.points[i])
                pois_times[labels[i]].append(datetime_list[i])
                pois_durations[labels[i]].append(durations_list[i])

        return pois_coordinates, pois_times, pois_durations