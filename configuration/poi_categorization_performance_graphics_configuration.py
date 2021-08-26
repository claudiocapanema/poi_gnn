from enum import Enum
import pytz


class PoiCategorizationPerformanceGraphicsConfiguration(Enum):

    MODELS_NAMES = ("models_names", ['gae',  'gcn', 'gat', 'gpr', 'arma'])

    BASE_DIR = ("base_dir", {'foursquare':
                                 {'directed': "output/performance_graphics/foursquare/directed/",
                                  'not_directed': "output/performance_graphics/foursquare/not_directed/"},
                             'weeplaces': {'directed': "output/performance_graphics/weeplaces/directed/",
                                           'not_directed': "output/performance_graphics/weeplaces/not_directed/"},
                             'global_foursquare': {'directed': "output/performance_graphics/global_foursquare/directed/",
                                                   'not_directed': {'BR': {'NORMAL': "output/performance_graphics/global_foursquare/BR/NORMAL/not_directed/",
                                                                           'PATH': "output/performance_graphics/global_foursquare/BR/PATH/not_directed/"},
                                                                    'US': {'NORMAL': "output/performance_graphics/global_foursquare/US/NORMAL/not_directed/",
                                                                           'PATH': "output/performance_graphics/global_foursquare/US/PATH/not_directed/"},
                                                                    'JP': "output/performance_graphics/global_foursquare/JP/not_directed/"}},
                             'dense_foursquare': {
                                 'directed': "output/performance_graphics/dense_foursquare/directed/",
                                 'not_directed': {'BR': {
                                     'normal': "output/performance_graphics/dense_foursquare/BR/normal/not_directed/",
                                     'PATH': "output/performance_graphics/dense_foursquare/BR/PATH/not_directed/"},
                                                  'US': {
                                                      'NORMAL': "output/performance_graphics/dense_foursquare/US/NORMAL/not_directed/",
                                                      'PATH': "output/performance_graphics/dense_foursquare/US/PATH/not_directed/"},
                                                  'JP': "output/performance_graphics/dense_foursquare/JP/not_directed/"}}
                             }
                )

    SEQUENTIAL_POI_RECOMMENDATION_BASELINES_MODELS_NAMES = ("sequential", ['map'])

    FOLDS_REPLICATIONS = ("folds_replications",
                          {'5': {'1': '5_folds/1_replications/',
                                 '2': '5_folds/2_replications/',
                                 '3': '5_folds/3_replications/'}})

    FOLDS_REPLICATIONS_FILENAME = ("folds_replications_filename",
                                   {'5': {'1': '5_folds_1_replications',
                                          '2': '5_folds_2_replications',
                                          '3': '5_folds_3_replications'}})

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.name

    def get_key(self):
        return self.value[0]

    def get_value(self):
        return self.value[1]