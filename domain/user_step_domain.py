import numpy as np
import pandas as pd

from foundation.util.geospatial_utils import points_distance
from foundation.util.datetimes_utils import DatetimesUtils
from extractor.file_extractor import FileExtractor
from model.confusion_matrix import ConfusionMatrix


class UserStepDomain:

    def __init__(self):
        self.file_extractor = FileExtractor()

    def users_steps_from_csv(self, filename):
        users_steps = self.file_extractor.read_csv(filename)
        # linha,id,datetime,latitude,longitude,handset,operating_system
        users_steps = users_steps[['id', 'local_date', 'latitude', 'longitude']]
        users_steps['id'] = users_steps['id'].astype('int64')
        users_steps['datetime'] = pd.to_datetime(users_steps['local_date'], infer_datetime_format=True)
        #users_steps = users_steps.drop_duplicates()
        users_steps = self._sort_users_records(users_steps)
        users_steps = self._calcule_users_durations(users_steps)
        return users_steps


    def _sort_users_records(self, users_steps):
        sorted_users_steps = pd.DataFrame(data={'id': [], 'datetime': [], 'latitude': [], 'longitude': []})
        ids = users_steps['id'].unique().tolist()

        for i in ids:
            user_records = users_steps.query("id==" + str(i))
            user_records = user_records.sort_values(by='datetime')
            sorted_users_steps = sorted_users_steps.append(user_records)

        return sorted_users_steps

    def _calcule_users_durations(self, users_steps):
        sorted_users_steps = pd.DataFrame(data={'id': [], 'datetime': [], 'latitude': [], 'longitude': []})
        ids = users_steps['id'].unique().tolist()
        durations = list()
        
        for i in ids:
            user_steps = users_steps.query(f"id=={i}")
            lat = user_steps["latitude"].tolist()
            long = user_steps["longitude"].tolist()
            datetime = user_steps["datetime"].tolist()
            
            user_durations = [0 for i in range(len(lat))]
            indice = 0
            for j in range(max(len(lat), len(long))):
                if(points_distance([lat[indice], long[indice]], [lat[j], long[j]]) > 100):
                    user_durations[j - 1] = DatetimesUtils.point_duration(datetime[indice], datetime[j - 1]) 
                    indice = j
                elif((j + 1) == len(lat)):
                    user_durations[j] = DatetimesUtils.point_duration(datetime[indice], datetime[j]) 
                    indice = j
            durations.extend(user_durations)
        users_steps["duration"] = durations

        return users_steps

    def user_pois_from_csv(self, filename):
        """ id, poi_type, latitude, longitude, work_time_events, home_time_events, inactive_interval_start,
        inactive_interval_end, inactive_applied_flag, inverted_routine_flag
        """
        df = self.file_extractor.read_csv(filename)
        # gdf = gpd.GeoDataFrame(
        #     df, geometry=gpd.points_from_xy(df.longitude, df.latitude))

        return df

    def ground_truth_from_csv(self, filename, poi_type_to_eng):
        """ID Usuario, Latitude, Longitude, Classe"""
        df = self.file_extractor.read_csv(filename)
        df.columns = ['id', 'latitude', 'longitude', 'poi_type']
        if(poi_type_to_eng == "yes"):
            transform = {"Casa": "home", "Trabalho": "work", "Outro": "other", "Lazer": "other"}
            df['poi_type'] = df['poi_type'].apply(lambda e: poi_type_to_eng[e])

        return df

    # def geodesic_point_buffer(lati, lon, km):
    #     # Azimuthal equidistant projection
    #     aeqd_proj = '+proj=aeqd +lat_0={lat} +lon_0={lon} +x_0=0 +y_0=0'
    #     proj_wgs84 = pyproj.Proj(init='epsg:4326')
    #     project = partial(
    #         pyproj.transform,
    #         pyproj.Proj(aeqd_proj.format(lat=lati, lon=lon)),
    #         proj_wgs84)
    #     buf = Point(0, 0).buffer(km * 1000)  # distance in metres
    #     return transform(project, buf).exterior.coords[:]


