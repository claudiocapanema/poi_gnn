import numpy as np
import pandas as pd

from domain.user_step_domain import UserStepDomain
from foundation.abs_classes.job import Job
from domain.matrix_generation_for_poi_categorization_domain import MatrixGenerationForPoiCategorizationDomain
from extractor.file_extractor import FileExtractor
from foundation.configuration.input import Input
from foundation.util.general_utils import join_df
from configuration.poi_categorization_configuration import PoICategorizationConfiguration
from configuration.matrix_generation_for_poi_categorization_configuration import MatrixGenerationForPoiCategorizationConfiguration
from loader.poi_categorization_loader import PoiCategorizationLoader
from pathlib import Path
from numba import jit

class MatrixGenerationForPoiCategorizationJob():

    def __init__(self):
        self.user_step_domain = UserStepDomain()
        self.file_extractor = FileExtractor()
        self.matrix_generation_for_poi_categorization_domain = MatrixGenerationForPoiCategorizationDomain(Input.get_instance().inputs['dataset_name'])
        self.poi_categorization_loader = PoiCategorizationLoader()
        self.poi_categorization_configuration = PoICategorizationConfiguration()

    @jit(nopython=True, parallel=True)
    def start(self):
        osm_category_column = None
        users_checkin_filename = Input.get_instance().inputs['users_checkin_filename']

        base_dir = Input.get_instance().inputs['base_dir']
        directed_folder = Input.get_instance().inputs['directed_folder']
        not_directed_folder = Input.get_instance().inputs['not_directed_folder']
        adjacency_matrix_base_filename = Input.get_instance().inputs['adjacency_matrix_base_filename']
        features_matrix_base_filename = Input.get_instance().inputs['features_matrix_base_filename']
        sequence_matrix_base_filename = Input.get_instance().inputs['sequence_matrix_base_filename']
        distance_matrix_base_filename = Input.get_instance().inputs['distance_matrix_base_filename']
        duration_matrix_base_filename = Input.get_instance().inputs['duration_matrix_base_filename']
        pattern_matrices = Input.get_instance().inputs['pattern_matrices']
        directed = Input.get_instance().inputs['directed']
        top_users = int(Input.get_instance().inputs['top_users'])
        dataset_name = Input.get_instance().inputs['dataset_name']
        categories_type = Input.get_instance().inputs['categories_type']
        personal_matrix = Input.get_instance().inputs['personal_features_matrix']
        hour48 = Input.get_instance().inputs['hour48']
        base = Input.get_instance().inputs['base']
        country = Input.get_instance().inputs['country']
        state = Input.get_instance().inputs['state']
        max_time_between_records = Input.get_instance().inputs['max_time_between_records']
        differemt_venues = Input.get_instance().inputs['different_venues']
        print("Dataset: ", Input.get_instance().inputs['dataset_name'])

        convert_country = {'Brazil': 'BR', 'BR': 'BR', 'United States': 'US'}

        if personal_matrix == "no":
            personal_matrix = False
        else:
            personal_matrix = True
        if hour48 == "no":
            hour48 = False
            hour_file = "24_"
        else:
            hour48 = True
            hour_file = "48_"

        country_folder = convert_country[country] + "/"
        if len(state) > 0:
            state_folder = state + "/"
        else:
            state_folder = ""

        different_venues_dir = ""
        if differemt_venues == "yes":
            different_venues_dir = "different_venues/"
            differemt_venues = True
        else:
            differemt_venues = False

        userid_column = MatrixGenerationForPoiCategorizationConfiguration.DATASET_COLUMNS.get_value()[dataset_name]['userid_column']
        category_column = MatrixGenerationForPoiCategorizationConfiguration.DATASET_COLUMNS.get_value()[dataset_name]['category_column']
        category_name_column = MatrixGenerationForPoiCategorizationConfiguration.DATASET_COLUMNS.get_value()[dataset_name]['category_name_column']
        locationid_column = MatrixGenerationForPoiCategorizationConfiguration.DATASET_COLUMNS.get_value()[dataset_name]['locationid_column']
        datetime_column = MatrixGenerationForPoiCategorizationConfiguration.DATASET_COLUMNS.get_value()[dataset_name]['datetime_column']
        latitude_column = MatrixGenerationForPoiCategorizationConfiguration.DATASET_COLUMNS.get_value()[dataset_name]['latitude_column']
        longitude_column = MatrixGenerationForPoiCategorizationConfiguration.DATASET_COLUMNS.get_value()[dataset_name]['longitude_column']
        country_column = MatrixGenerationForPoiCategorizationConfiguration.DATASET_COLUMNS.get_value()[dataset_name]['country_column']
        state_column = MatrixGenerationForPoiCategorizationConfiguration.DATASET_COLUMNS.get_value()[dataset_name]['state_column']
        num_users = MatrixGenerationForPoiCategorizationConfiguration.NUM_USERS.get_value()[dataset_name]
        category_to_int = self.poi_categorization_configuration.CATEGORIES_TO_INT[dataset_name][categories_type]
        max_time_between_records_dir = self.poi_categorization_configuration.MAX_TIME_BETWEEN_RECORDS[1][max_time_between_records]

        # get list of valid categories for the given dataset
        # categories_to_int_osm = self.poi_categorization_configuration.\
        #     DATASET_CATEGORIES_TO_INT_OSM_CATEGORIES[1][dataset_name][categories_type]
        max_size_matrices = self.poi_categorization_configuration.MAX_SIZE_MATRICES[1]
        n_splits = self.poi_categorization_configuration.N_SPLITS[1]
        n_replications = self.poi_categorization_configuration.N_REPLICATIONS[1]

        output_base_dir = self.poi_categorization_configuration.OUTPUT_DIR[1]
        dataset_type_dir = self.poi_categorization_configuration.DATASET_TYPE[1][dataset_name]
        category_type_dir = self.poi_categorization_configuration.CATEGORY_TYPE[1][categories_type]
        output_dir = output_base_dir + dataset_type_dir + category_type_dir

        dtypes_columns = {userid_column: int, category_column: 'Int16', category_name_column: 'category',
                          locationid_column: 'category', datetime_column: 'category', latitude_column: 'float64',
                          longitude_column: 'float64'}

        print(dtypes_columns)

        #base_report = self.poi_categorization_configuration.REPORT_MODEL[1][categories_type]
        users_checkin = self.file_extractor.read_csv(users_checkin_filename, dtypes_columns).query(country_column + " == '"+country+"'")
        if category_column == category_name_column:
            categories = users_checkin[category_name_column].tolist()
            categories_int = []
            for i in range(len(categories)):
                categories_int.append(category_to_int[categories[i]])
            category_column = category_column + "_id"
            users_checkin[category_column] = np.array(categories_int)

        if state != "":
            users_checkin = users_checkin.query(state_column + " == '" + state + "'")
        print("----- verificação -----")
        print("Pais: ", users_checkin[country_column].unique().tolist())
        if len(state) > 0:
            print("Estado: ", users_checkin[state_column].unique().tolist())

        # data

        users_checkin[datetime_column] = pd.to_datetime(users_checkin[datetime_column], infer_datetime_format=True)
        users_checkin[category_column] = users_checkin[category_column].astype('int')
        # min_datetime = pd.Timestamp(year=2012, month=4, day=1, tz='utc')
        # max_datetime = pd.Timestamp(year=2013, month=10, day=1, tz='utc')
        # users_checkin = users_checkin[((users_checkin['local_datetime']< max_datetime) & (users_checkin['local_datetime']>= min_datetime))]
        # country = "from_2012_month_4_to_2013_month_9_" + country
        # directory = "/media/claudio/Data/backup_win_hd/Downloads/doutorado/global_foursquare/dataset_TIST2015/comparar_csv/"
        # users_checkin.to_csv(directory+"from_2012_2013_to_2012.csv", index_label=False, index=False)
        if dataset_name == 'raw_gps':
            personal_category_column = MatrixGenerationForPoiCategorizationConfiguration.DATASET_COLUMNS.get_value()[dataset_name]['personal_category_column']
            users_checkin = users_checkin.query("" + personal_category_column + " != 'home'")
            # coluna com as categorias em um determinado raio em metros
            osm_category_column = MatrixGenerationForPoiCategorizationConfiguration.DATASET_COLUMNS.get_value()[dataset_name]['osm_category_column']

        print("coluna osm", osm_category_column)
        #----------------------

        """
        Generate matrixes for each user 
        """
        # if len(max_time_between_records) > 0:
        #     max_time_between_records = max_time_between_records + "/"
        if personal_matrix:
            directed = False
            folder = base_dir + base + "/" + not_directed_folder + country_folder + state_folder + max_time_between_records_dir
            adjacency_matrix_base_filename = folder + adjacency_matrix_base_filename + "not_directed_personal_" + hour_file + categories_type + ".csv"
            features_matrix_base_filename = folder + features_matrix_base_filename + "not_directed_personal_" + hour_file + categories_type + ".csv"
            sequence_matrix_base_filename = folder + sequence_matrix_base_filename + "not_directed_personal_" + hour_file + categories_type + ".csv"
        elif directed == "no":
            directed = False
            folder = base_dir + different_venues_dir + base + "/" + not_directed_folder + country_folder + state_folder + max_time_between_records_dir
            print("Pasta: ", folder)
            self.folder_generation(folder)
            # features_matrix_base_filename = folder+features_matrix_base_filename+"not_directed_"+hour_file+categories_type+"_"+country+".csv"
            # sequence_matrix_base_filename = folder+sequence_matrix_base_filename+"not_directed_"+hour_file+categories_type+"_"+country+".csv"

            country = convert_country[country]
            adjacency_matrix_filename = folder + adjacency_matrix_base_filename + "_not_directed_" + hour_file + categories_type + "_" + country + ".csv"
            print("nome matriz", adjacency_matrix_filename)
            adjacency_weekday_matrix_filename = folder + adjacency_matrix_base_filename + "_weekday_not_directed_"+hour_file+categories_type+"_"+country+".csv"
            adjacency_weekend_matrix_filename = folder+adjacency_matrix_base_filename + "_weekend_not_directed_"+hour_file+categories_type+"_"+country+".csv"
            temporal_matrix_filename = folder+features_matrix_base_filename + "_not_directed_"+hour_file+categories_type+"_"+country+".csv"
            temporal_weekday_matrix_filename = folder+features_matrix_base_filename + "_weekday_not_directed_"+hour_file+categories_type+"_"+country+".csv"
            temporal_weekend_matrix_filename = folder+features_matrix_base_filename + "_weekend_not_directed_"+hour_file+categories_type+"_"+country+".csv"
            path_matrix_filename = folder+sequence_matrix_base_filename + "_not_directed_"+hour_file+categories_type+"_"+country+".csv"
            path_weekeday_matrix_filename = folder+sequence_matrix_base_filename + "_weekday_not_directed_"+hour_file+categories_type+"_"+country+".csv"
            path_weekend_matrix_filename = folder+sequence_matrix_base_filename + "_weekend_not_directed_"+hour_file+categories_type+"_"+country+".csv"
            distance_matrix_filename = folder + distance_matrix_base_filename + "_not_directed_" + hour_file + categories_type + "_" + country + ".csv"
            distance_weekday_matrix_filename = folder + distance_matrix_base_filename + "_weekday_not_directed_" + hour_file + categories_type + "_" + country + ".csv"
            distance_weekend_matrix_filename = folder + distance_matrix_base_filename + "_weekend_not_directed_" + hour_file + categories_type + "_" + country + ".csv"
            duration_matrix_filename = folder + duration_matrix_base_filename + "_not_directed_" + hour_file + categories_type + "_" + country + ".csv"
            duration_weekday_matrix_filename = folder + duration_matrix_base_filename + "_weekday_not_directed_" + hour_file + categories_type + "_" + country + ".csv"
            duration_weekend_matrix_filename = folder + duration_matrix_base_filename + "_weekend_not_directed_" + hour_file + categories_type + "_" + country + ".csv"
            location_locaion_pmi_matrix_filename = folder + "location_location_pmi_matrix_" + categories_type + "_" + country + ".npz"
            location_time_pmi_matrix_filename = folder + "location_time_pmi_matrix_" + categories_type + "_" + country + ".csv"
            int_to_locationid_filename = folder + "int_to_locationid_" + categories_type + "_" + country + ".csv"
        else:
            directed = True
            folder = base_dir + directed_folder
            adjacency_matrix_base_filename = folder+adjacency_matrix_base_filename + "directed_"+hour_file+categories_type+"_"+country+".csv"
            features_matrix_base_filename = folder+features_matrix_base_filename + "directed_"+hour_file+categories_type+"_"+country+".csv"
            sequence_matrix_base_filename = folder + sequence_matrix_base_filename + "directed_" + hour_file + categories_type +"_"+country+ ".csv"

        print("arquivos: ", folder, adjacency_matrix_base_filename, features_matrix_base_filename)

        print("padrao", pattern_matrices)
        print("tamanho: ", users_checkin.shape)
        if pattern_matrices == "yes":
            self.matrix_generation_for_poi_categorization_domain\
                .generate_pattern_matrices(users_checkin,
                                           dataset_name,
                                           adjacency_matrix_filename,
                                           adjacency_weekday_matrix_filename,
                                           adjacency_weekend_matrix_filename,
                                           temporal_matrix_filename,
                                           temporal_weekday_matrix_filename,
                                           temporal_weekend_matrix_filename,
                                           distance_matrix_filename,
                                           duration_matrix_filename,
                                           location_locaion_pmi_matrix_filename,
                                           location_time_pmi_matrix_filename,
                                           int_to_locationid_filename,
                                           userid_column,
                                           category_column,
                                           locationid_column,
                                           latitude_column,
                                           longitude_column,
                                           datetime_column,
                                           differemt_venues,
                                           directed,
                                           personal_matrix,
                                           top_users,
                                           max_time_between_records,
                                           num_users,
                                           hour48,
                                           osm_category_column)
        else:
            self.matrix_generation_for_poi_categorization_domain \
                .generate_gpr_matrices(users_checkin,
                                           adjacency_matrix_base_filename,
                                           features_matrix_base_filename,
                                           userid_column,
                                           latitude_column,
                                           longitude_column,
                                           category_column,
                                           locationid_column,
                                           datetime_column,
                                           directed,
                                           osm_category_column)

    def folder_generation(self, folder):
        print("criação da pas: ", folder)
        Path(folder).mkdir(parents=True, exist_ok=True)