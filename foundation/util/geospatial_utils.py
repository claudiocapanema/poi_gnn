import pandas as pd
from sklearn.metrics.pairwise import haversine_distances
import numpy as np

def points_distance(point_0, point_1):
    """
    :param point_0: [lat, lng]
    :param point_1: [lat, lng]
    :return: distance
    """
    point_0 = np.radians(point_0)
    point_1 = np.radians(point_1)
    result = haversine_distances([point_0, point_1])
    # metros
    result = result * 6371000
    distance = result[0][1]
    return distance

def centroid(latitudes, longitudes):
    lenght = len(latitudes)
    sum_lat = sum(latitudes)
    sum_long = sum(longitudes)

    latitude = sum_lat / lenght
    longitude = sum_long / lenght

    return latitude, longitude