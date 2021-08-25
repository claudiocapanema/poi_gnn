#!/usr/bin/env python
# coding: utf-8
import pandas as pd

class User:

    def __init__(self, id_, pois, n_events, meters, min_samples, min_home_events, min_work_events):
        # app user id
        self.id = id_
        self.pois = pois
        self.home_correct = False
        self.work_correct = False
        self.home_entropy = None
        self.work_entropy = None
        self.total_home_events = 0
        self.total_work_events = 0
        self.min_home_events = min_home_events
        self.min_work_events = min_work_events
        self.n_events = n_events
        self.meters = meters
        self.min_samples = min_samples
        self.inactive_interval_start = -1
        self.inactive_interval_end = -1
        self.inactive_applied_flag = str(False)
        self.inverted_routine_flag = str(False)

    def calculate_total_home_work_events(self):

        for poi in self.pois:
            self.total_home_events = self.total_home_events + poi.total_home_events
            self.total_work_events = self.total_work_events + poi.total_work_events

    def calculate_different_days(self):
        for i in range(len(self.pois)):
            self.pois[i].calculate_different_days()

    def user_pois_to_pandas_df(self):
        number_of_pois = len(self.pois)
        ids = [self.id]*number_of_pois
        locations_types = []
        latitudes = []
        longitudes = []
        homes_times_events = []
        works_times_events = []
        inactive_interval_start = [self.inactive_interval_start] * number_of_pois
        inactive_interval_end = [self.inactive_interval_end] * number_of_pois
        inactive_applied_flag = [self.inactive_applied_flag] * number_of_pois
        inverted_routine_flag = [self.inverted_routine_flag] * number_of_pois
        for poi in self.pois:
            poi = poi.to_dict()
            locations_types.append(poi['location_type'])
            latitudes.append(poi['latitude'])
            longitudes.append(poi['longitude'])
            homes_times_events.append(poi['home_time_events'])
            works_times_events.append(poi['work_time_events'])

        df = pd.DataFrame({"id": ids, "poi_type": locations_types, "latitude": latitudes, "longitude": longitudes,
                           "work_time_events": works_times_events, "home_time_events": homes_times_events,
                           "inactive_interval_start": inactive_interval_start,
                           "inactive_interval_end": inactive_interval_end, "inactive_applied_flag": inactive_applied_flag,
                           "inverted_routine_flag": inverted_routine_flag})

        return df
