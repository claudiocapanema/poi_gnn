#!/usr/bin/python
# -*- coding: utf-8 -*-

from copy import copy
import numpy as np
from configuration.weekday import Weekday

class Poi:

    def __init__(self, coordinates, times, durations):
        self._centroid = self.find_centroid(coordinates)
        self._coordinates = coordinates
        self._n_events = len(coordinates)
        self._coord_per_day = []
        self._times = times
        self._different_hours = None
        self._durations = durations
        self._duration_home_time = 0
        self._duration_work_time = 0
        self._n_events_week = None
        self._poi_duration = None
        self._poi_class = "other"
        self._home_hour = {}
        self._work_hour = {}
        self._different_days = 0
        self._different_schedules = 0
        self._n_events_work_time = 0
        self._n_events_home_time = 0
        self._n_events_week = 0
        self._n_events_weekend = 0
        self._calculate_different_days()
        self._calculate_different_hours()
        self._calculate_total_duration()
    

    def mean_points_per_day(self):
        points_per_day = []
        points_counter = 0
        day = self.times[0].day + self.times[0].month

        for i in self.times:
            if((i.day + i.month) != day):
                day = i.day + i.month
                points_per_day.append(points_counter)
                points_counter = 1
            points_counter += 1
        points_per_day.append(points_counter)
        return np.mean(points_per_day)

    def change_first_and_last_days_coords(self):
        first_and_last_days_coords = {}
        times_sorted = [x for x in self.times]
        times_sorted.sort()

        day = f"{times_sorted[0].day}/{times_sorted[0].month}"
        first_point = times_sorted[0].day
        for i in range(len(times_sorted)): 
            if(times_sorted[i].day != first_point.day):
                last_point = times_sorted[i - 1]
                first_and_last_days_coords[day] = (first_point, last_point)
                
                day = f"{times_sorted[i].day}/{times_sorted[i].month}"
                first_point = times_sorted[i]
        
        last_point = times_sorted[len(times_sorted) - 1]
        first_and_last_days_coords[day] = (first_point, last_point)
        return first_and_last_days_coords



    def _calculate_total_duration(self):
        """
           Method to calculate the total duration of event.

        """

        durations = self._durations
        total_durations = 0
        for i in range(len(durations)):
            total_durations += durations[i]
        self._poi_duration = total_durations

    def _calculate_different_hours(self):
        """
           Method to calculate the number of different hours that the events occurred.

        """
        self._times
        events_hour = [0] * 24
        for i in range(len(self.times)):
            events_hour[self.times[i].hour] = events_hour[self.times[i].hour] + 1
        self._different_hours = events_hour
        self._different_schedules = np.count_nonzero(events_hour)

    def _calculate_different_days(self):
        """
               Method to calculate the number of different days that the events occurred.

       """
        times = self._times
        events_day = [0] * 32
        different_days = 0
        for i in range(len(times)):
            events_day[times[i].day] = events_day[times[i].day] + 1
        for i in range(len(events_day)):
            if events_day[i] != 0:
                different_days = different_days + 1
        self._different_days =  different_days
    
    @property
    def poi_duration(self):
        return self._poi_duration

    @property
    def different_hours(self):
        return self._different_hours

    @property
    def n_events_week(self):
        return self._n_events_week

    @property
    def n_events_weekend(self):
        return self._n_events_weekend

    @property
    def different_days(self):
        return self._different_days

    @property
    def n_events(self):
        return self._n_events

    @property
    def different_schedules(self):
        return self._different_schedules

    def __repr__(self):
        return "Centroide:"+str(self._centroid)

    @property
    def centroid(self):
        return self._centroid

    @property
    def duration_home_time(self):
        return self._duration_home_time

    @property
    def duration_work_time(self):
        return self._duration_work_time

    @property
    def n_events_work_time(self):
        return self._n_events_work_time

    @property
    def n_events_home_time(self):
        return self._n_events_home_time

    @property
    def times(self):
        return self._times

    @property
    def poi_class(self):
        return self._poi_class

    @poi_class.setter
    def poi_class(self, poi_class):
        self._poi_class = poi_class

    @property
    def home_hour(self):
        return self._home_hour

    @property
    def work_hour(self):
        return self._work_hour

    @home_hour.setter
    def home_hour(self, home_hour):
        self._home_hour = home_hour

    @work_hour.setter
    def work_hour(self, work_hour):
        self._work_hour = work_hour

    def add_n_events_home_time(self):
        self._n_events_home_time = self._n_events_home_time +1

    def add_n_events_work_time(self):
        self._n_events_work_time = self._n_events_work_time +1
    
    def add_duration_home_time(self, index):
        self._duration_home_time += self._durations[index]

    def add_duration_work_time(self, index):
        self._duration_work_time += self._durations[index]

    def add_n_events_week(self):
        self._n_events_week = self._n_events_week +1

    def add_n_events_weekend(self):
        self._n_events_weekend = self._n_events_weekend +1

    def to_dict(self):
        return {'location_type': self._poi_class, 'latitude': str(self.centroid[0]), 'longitude': str(self.centroid[1]), \
                 'home_time_events': str(self.n_events_home_time), 'work_time_events': str(self.n_events_work_time)}

    def calculate_n_events(self):
        times = self._times
        for i in range(len(times)):
            if times[i].weekday() < 5:
                self.add_n_events_week()
            else:
                self.add_n_events_weekend()

    def calculate_home_work_n_events(self):

        for datetime in self.times:
            if self.home_hour['start'] < self.home_hour['end']:
                if datetime.hour >= self.home_hour['start'] and datetime.hour <= self.home_hour['end']:
                    self.add_n_events_home_time()
            else:
                if datetime.hour >= self.home_hour['start'] or datetime.hour <= self.home_hour['end']:
                    self.add_n_events_home_time()
            # steps generated on weekends are not accounted for finding the work POI
            if datetime.weekday() >= Weekday.SATURDAY.value:
                continue
            elif self.work_hour['start'] < self.work_hour['end']:
                if datetime.hour >= self.work_hour['start'] and datetime.hour <= self.work_hour['end']:
                    self.add_n_events_work_time()
            else:
                if datetime.hour >= self.work_hour['start'] or datetime.hour <= self.work_hour['end']:
                    self.add_n_events_work_time()

    def calculate_home_and_work_durations(self):
        for i in range(len(self.times)):
            if self.home_hour['start'] < self.home_hour['end']:
                if self.times[i].hour >= self.home_hour['start'] and self.times[i].hour <= self.home_hour['end']:
                    self.add_duration_home_time(i)
            else:
                if self.times[i].hour >= self.home_hour['start'] or self.times[i].hour <= self.home_hour['end']:
                    self.add_duration_home_time(i)
            # steps generated on weekends are not accounted for finding the work POI
            if self.times[i].weekday() >= Weekday.SATURDAY.value:
                continue
            elif self.work_hour['start'] < self.work_hour['end']:
                if self.times[i].hour >= self.work_hour['start'] and self.times[i].hour <= self.work_hour['end']:
                    self.add_duration_work_time(i)
            else:
                if self.times[i].hour >= self.work_hour['start'] or self.times[i].hour <= self.work_hour['end']:
                    self.add_duration_work_time(i)

    def find_centroid(self, vertexes):
        _x_list = [vertex[0] for vertex in vertexes]
        _y_list = [vertex[1] for vertex in vertexes]
        _len = len(vertexes)
        _x = round(sum(_x_list) / _len,8)
        _y = round(sum(_y_list) / _len,8)
        return (_x, _y)