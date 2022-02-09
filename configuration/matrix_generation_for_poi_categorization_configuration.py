from enum import Enum
import pytz


class MatrixGenerationForPoiCategorizationConfiguration(Enum):

    # Radius for the nearestneighbors algorithm - 100m

    CATEGORIES = ("radius", {"9": "_9categories", "12": "_12categories"}, False, "radius in meters for the nearestneighbors alogrithm")
    DATASET_COLUMNS = ("dataset_columns",
                       {'weeplaces':
                            {'userid_column':'userid',
                            'category_column':'category',
                            'locationid_column':'placeid',
                            'datetime_column':'local_datetime',
                             'latitude_column': 'lat',
                             'longitude_column': 'lon'},
                        'foursquare':
                            {'userid_column':'',
                            'category_column':'',
                            'locationid_column':'',
                            'datetime_column':''},
                        'user_tracking': {"datetime_column": "datetime",
                                          "userid_column": "id",
                                            "locationid_column": "poi_id",
                                            "country_column": "country_name",
                                            "state_column": "state_name",
                                            "category_column": "poi_resulting",
                                            "category_name_column": "poi_resulting",
                                            "latitude_column": "latitude",
                                            "longitude_column": "longitude"},
                        "global_foursquare": {"datetime_column": "local_datetime",
                                              "userid_column": "userid",
                                              "locationid_column": "placeid",
                                              "country_column": "country_code",
                                              "category_column": "categoryid",
                                              "category_name_column": "category",
                                              "latitude_column": "latitude",
                                              "longitude_column": "longitude"},
                        "dense_foursquare": {"datetime_column": "local_datetime",
                                              "userid_column": "userid",
                                              "locationid_column": "placeid",
                                             "country_column": "country_name",
                                             "state_column": "state_name",
                                              "category_column": "categoryid",
                                              "category_name_column": "category",
                                              "latitude_column": "latitude",
                                              "longitude_column": "longitude"},
                        "gowalla": {"datetime_column": "local_datetime",
                                              "userid_column": "userid",
                                              "locationid_column": "placeid",
                                             "country_column": "country_name",
                                            "state_column": "state_name",
                                              "category_column": "category",
                                              "category_name_column": "category",
                                              "latitude_column": "latitude",
                                              "longitude_column": "longitude"},
                         })

    NUM_USERS = ("num_users", {'dense_foursquare': 30000,
                               'gowalla': 2000,
                               'user_tracking': 10000})

    # userid	placeid	datetime	lat	lon	city	category	local_datetime

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.name

    def get_key(self):
        return self.value[0]

    def get_value(self):
        return self.value[1]