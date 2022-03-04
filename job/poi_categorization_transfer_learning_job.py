import numpy as np
import pandas as pd

from domain.user_step_domain import UserStepDomain
from foundation.abs_classes.job import Job
from domain.poi_categorization_transfer_learning_domain import PoiCategorizationTransferLearningDomain
from extractor.file_extractor import FileExtractor
from foundation.configuration.input import Input
from foundation.util.general_utils import join_df
from configuration.poi_categorization_configuration import PoICategorizationConfiguration
from loader.poi_categorization_loader import PoiCategorizationLoader

class PoiCategorizationTransferLearningJob:

    def __init__(self):
        self.user_step_domain = UserStepDomain()
        self.file_extractor = FileExtractor()
        self.transfer_learning_domain = PoiCategorizationTransferLearningDomain(Input.get_instance().inputs['dataset_name'])
        self.poi_categorization_loader = PoiCategorizationLoader()
        self.poi_categorization_configuration = PoICategorizationConfiguration()

    def start(self):
        base_dir = Input.get_instance().inputs['base_dir']
        adjacency_matrix_filename = Input.get_instance().inputs['adjacency_matrix_filename']
        adjacency_matrix_week_filename = Input.get_instance().inputs['adjacency_matrix_week_filename']
        adjacency_matrix_weekend_filename = Input.get_instance().inputs['adjacency_matrix_weekend_filename']
        graph_type = Input.get_instance().inputs['graph_type']
        temporal_matrix_filename = Input.get_instance().inputs['temporal_matrix_filename']
        temporal_matrix_week_filename = Input.get_instance().inputs['temporal_matrix_week_filename']
        temporal_matrix_weekend_filename = Input.get_instance().inputs['temporal_matrix_weekend_filename']
        distance_matrix_filename = Input.get_instance().inputs['distance_matrix_filename']
        # distance_matrix_weekday_filename = Input.get_instance().inputs['distance_matrix_week_filename']
        # distance_magrix_weekend_filename = Input.get_instance().inputs['distance_matrix_weekend_filename']
        duration_matrix_filename = Input.get_instance().inputs['duration_matrix_filename']
        # duration_matrix_weekday_filename = Input.get_instance().inputs['duration_matrix_week_filename']
        # duration_matrix_weekend_filename = Input.get_instance().inputs['duration_matrix_weekend_filename']
        dataset_name = Input.get_instance().inputs['dataset_name']
        base = Input.get_instance().inputs['base']
        categories_type = Input.get_instance().inputs['categories_type']
        location_location_filename = Input.get_instance().inputs['location_location_filename']
        location_time_filename = Input.get_instance().inputs['location_time_filename']
        int_to_locationid_filename = Input.get_instance().inputs['int_to_locationid_filename']
        country = Input.get_instance().inputs['country']
        state = Input.get_instance().inputs['state']
        version = Input.get_instance().inputs['version']
        print("Dataset: ", Input.get_instance().inputs['dataset_name'])

        # assert tf.test.is_gpu_available()
        # assert tf.test.is_built_with_cuda()

        max_size_matrices = self.poi_categorization_configuration.MAX_SIZE_MATRICES[1]
        max_size_paths = self.poi_categorization_configuration.MINIMUM_RECORDS[1]
        n_splits = self.poi_categorization_configuration.N_SPLITS[1]
        n_replications = self.poi_categorization_configuration.N_REPLICATIONS[1]
        epochs = self.poi_categorization_configuration.EPOCHS[1][country]
        print("contar", self.poi_categorization_configuration.INT_TO_CATEGORIES[1][dataset_name])
        print("ori", self.poi_categorization_configuration.INT_TO_CATEGORIES[1])
        output_base_dir = self.poi_categorization_configuration.OUTPUT_DIR[1]
        dataset_type_dir = self.poi_categorization_configuration.DATASET_TYPE[1][dataset_name]
        category_type_dir = self.poi_categorization_configuration.CATEGORY_TYPE[1][categories_type]
        int_to_category = self.poi_categorization_configuration.INT_TO_CATEGORIES[1][dataset_name][categories_type]
        graph_type_dir = self.poi_categorization_configuration.GRAPH_TYPE[1][graph_type]
        country_dir = self.poi_categorization_configuration.COUNTRY[1][country]
        state_dir = self.poi_categorization_configuration.STATE[1][state]
        version_dir = self.poi_categorization_configuration.VERSION[1][version]
        if len(base) > 0:
            base = base + "/"
        output_dir = self.poi_categorization_configuration.output_dir(output_base_dir=output_base_dir,
                                                                      base=base,
                                                                      graph_type=graph_type_dir,
                                                                      dataset_type=dataset_type_dir,
                                                                      country=country_dir,
                                                                      category_type=category_type_dir,
                                                                      version=version_dir,
                                                                      state_dir=state_dir,
                                                                      max_time_between_records_dir="")


        base_report = self.poi_categorization_configuration.REPORT_MODEL[1][categories_type]
        max_time_between_records_dir = ""

        if len(state) > 0:
            base_dir = base_dir + base + graph_type + "/" + country_dir + state + "/" + max_time_between_records_dir
        else:
            base_dir = base_dir + base + graph_type + "/" + country_dir + max_time_between_records_dir

        if len(base) > 0:
            base = True
        else:
            base = False

        adjacency_matrix_filename = base_dir + adjacency_matrix_filename
        temporal_matrix_filename = base_dir + temporal_matrix_filename
        distance_matrix_filename = base_dir + distance_matrix_filename
        duration_matrix_filename = base_dir + duration_matrix_filename
        adjacency_matrix_week_filename = base_dir + adjacency_matrix_week_filename
        temporal_matrix_week_filename = base_dir + temporal_matrix_week_filename
        adjacency_matrix_weekend_filename = base_dir + adjacency_matrix_weekend_filename
        temporal_matrix_weekend_filename = base_dir + temporal_matrix_weekend_filename

        self.files_verification(country, state, adjacency_matrix_filename, temporal_matrix_filename,
                           adjacency_matrix_week_filename, temporal_matrix_week_filename,
                           adjacency_matrix_weekend_filename, temporal_matrix_weekend_filename,
                           distance_matrix_filename,
                           duration_matrix_filename)

        # normal matrices
        adjacency_df, temporal_df, distance_df, duration_df = self.transfer_learning_domain.\
            read_matrix(adjacency_matrix_filename, temporal_matrix_filename, distance_matrix_filename, duration_matrix_filename)
        print("arquivos: \n", adjacency_matrix_filename)
        print(adjacency_df)
        print(temporal_matrix_filename)

        # week matrices
        adjacency_week_df, temporal_week_df, distance_week_df, duration_week_df = self.transfer_learning_domain. \
            read_matrix(adjacency_matrix_week_filename, temporal_matrix_week_filename)
        # weekend matrices
        adjacency_weekend_df, temporal_weekend_df, distance_weekend_df, duration_weekend_df = self.transfer_learning_domain. \
            read_matrix(adjacency_matrix_weekend_filename, temporal_matrix_weekend_filename)
        print("Verificação de matrizes")
        self.matrices_verification(adjacency_df, temporal_df, adjacency_week_df, temporal_week_df,
                                   adjacency_weekend_df, temporal_weekend_df, distance_df, duration_df)

        location_location = self.file_extractor.read_npz(base_dir +location_location_filename)
        location_time = self.file_extractor.read_csv(base_dir + location_time_filename)
        int_to_locationid = self.file_extractor.read_csv(base_dir + int_to_locationid_filename)
        inputs = {'all_week': {'adjacency': adjacency_df, 'temporal': temporal_df, 'distance': distance_df, 'duration': duration_df,
                               'location_location': location_location, 'location_time': location_time, 'int_to_locationid': int_to_locationid},
                  'week': {'adjacency': adjacency_week_df, 'temporal': temporal_week_df},
                  'weekend': {'adjacency': adjacency_weekend_df, 'temporal': temporal_weekend_df}}

        print("Preprocessing")
        users_categories, adjacency_df, temporal_df, distance_df, duration_df, adjacency_week_df, temporal_week_df,\
        adjacency_weekend_df, temporal_weekend_df, location_time_df, location_location_df, selected_users, df_selected_users_visited_locations\
            = self.transfer_learning_domain.adjacency_preprocessing(inputs,
                                    max_size_matrices,
                                    True,
                                    True,
                                    7,
                                    dataset_name)

        selected_users = pd.DataFrame({'selected_users': selected_users})

        self.matrices_verification(adjacency_df, temporal_df, adjacency_week_df, temporal_week_df,
                              adjacency_weekend_df, temporal_weekend_df, distance_df, distance_week_df)

        usuarios = len(adjacency_df)

        inputs = users_categories, adjacency_df, temporal_df, distance_df, duration_df, adjacency_week_df, temporal_week_df,\
        adjacency_weekend_df, temporal_weekend_df, location_time_df, location_location_df

        print("Treino")
        y_predicted = self.transfer_learning_domain.\
            k_fold_with_replication_train_and_evaluate_model(inputs,
                                                             max_size_matrices,
                                                             base_report,
                                                             epochs,
                                                             base,
                                                             country,
                                                             version,
                                                             output_dir)

        df_selected_users_visited_locations['category'] = np.array([int_to_category[str(i)] for i in y_predicted])
        df_selected_users_visited_locations['category_id'] = np.array(y_predicted)

        df_selected_users_visited_locations.to_csv(
            "/media/claudio/Data/backup_win_hd/Downloads/doutorado/users_steps_output/users_steps_10_mil_limite_500_pontos_local_datetime_with_detected_pois_with_osm_pois_50_gowalla_categories_transfer_learning_predict_br.csv",
            index=False)

        selected_users.to_csv(output_dir + "selected_users.csv", index=False)
        print("base: ", base_dir)
        print("------------- Location ------------")
        print(y_predicted)


    def files_verification(self, country, state, adjacency_matrix_filename, temporal_matrix_filename,
                           adjacency_matrix_week_filename, temporal_matrix_week_filename,
                           adjacency_matrix_weekend_filename, temporal_matrix_weekend_filename,
                           distance_matrix_filename,
                           duration_matrix_filename):

        if country not in adjacency_matrix_filename or country not in temporal_matrix_filename \
                or country not in adjacency_matrix_week_filename or country not in adjacency_matrix_weekend_filename or country not in \
                distance_matrix_filename:

            print("matrizes diferentes do país")
            print(adjacency_matrix_filename)
            print(temporal_matrix_filename)
            print(adjacency_matrix_week_filename)
            print(temporal_matrix_week_filename)
            print(adjacency_matrix_weekend_filename)
            print(temporal_matrix_weekend_filename)

        if state not in adjacency_matrix_filename or state not in temporal_matrix_filename \
                or state not in \
                adjacency_matrix_week_filename or state not in adjacency_matrix_weekend_filename or state not in \
                temporal_matrix_week_filename or state not in temporal_matrix_weekend_filename:
            print("matrizes diferentes do estado")
            print(adjacency_matrix_filename)
            print(temporal_matrix_filename)
            print(adjacency_matrix_week_filename)
            print(temporal_matrix_week_filename)
            print(adjacency_matrix_weekend_filename)
            print(temporal_matrix_weekend_filename)
            raise

        if 'week' not in adjacency_matrix_week_filename or 'week' not in temporal_matrix_week_filename:
            print("matrizes diferentes de week")
            print(adjacency_matrix_week_filename)
            print(temporal_matrix_week_filename)
            raise

        if 'weekend' not in adjacency_matrix_weekend_filename or 'weekend' not in temporal_matrix_weekend_filename:
            print("matrizes diferentes weekend")
            print(adjacency_matrix_weekend_filename)
            print(temporal_matrix_weekend_filename)
            raise

    def matrices_verification(self, adjacency_df, temporal_df, adjacency_week_df, temporal_week_df,
                              adjacency_weekend_df, temporal_weekend_df,  distance_df, duration_df):

        if not(len(adjacency_df) == len(temporal_df) == len(adjacency_week_df) == len(temporal_week_df) == len(adjacency_weekend_df) == len(temporal_weekend_df) == len(distance_df)):
            print("Matrizes com tamanhos diferentes")
            raise
        else:
            print("Quantidade inicial de usuários: ", len(adjacency_df))