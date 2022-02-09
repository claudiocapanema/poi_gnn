import numpy as np
import pandas as pd
import tensorflow as tf

from domain.user_step_domain import UserStepDomain
from foundation.abs_classes.job import Job
from domain.poi_categorization_domain import PoiCategorizationDomain
from extractor.file_extractor import FileExtractor
from foundation.configuration.input import Input
from foundation.util.general_utils import join_df
from configuration.poi_categorization_configuration import PoICategorizationConfiguration
from loader.poi_categorization_loader import PoiCategorizationLoader

class PoiCategorizationJob:

    def __init__(self):
        self.user_step_domain = UserStepDomain()
        self.file_extractor = FileExtractor()
        self.poi_categorization_domain = PoiCategorizationDomain(Input.get_instance().inputs['dataset_name'])
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
        distance_matrix_weekday_filename = Input.get_instance().inputs['distance_matrix_week_filename']
        distance_magrix_weekend_filename = Input.get_instance().inputs['distance_matrix_weekend_filename']
        duration_matrix_filename = Input.get_instance().inputs['duration_matrix_filename']
        duration_matrix_weekday_filename = Input.get_instance().inputs['duration_matrix_week_filename']
        duration_matrix_weekend_filename = Input.get_instance().inputs['duration_matrix_weekend_filename']
        dataset_name = Input.get_instance().inputs['dataset_name']
        base = Input.get_instance().inputs['base']
        categories_type = Input.get_instance().inputs['categories_type']
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
            base_dir = base_dir + graph_type + "/" + country_dir + state + "/" + max_time_between_records_dir
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
        distance_matrix_weekday_filename = base_dir + distance_matrix_weekday_filename
        duration_matrix_weekday_filename = base_dir + duration_matrix_weekday_filename
        adjacency_matrix_weekend_filename = base_dir + adjacency_matrix_weekend_filename
        temporal_matrix_weekend_filename = base_dir + temporal_matrix_weekend_filename
        distance_magrix_weekend_filename = base_dir + distance_magrix_weekend_filename
        duration_matrix_weekend_filename = base_dir + duration_matrix_weekend_filename

        self.files_verification(country, state, adjacency_matrix_filename, temporal_matrix_filename,
                           adjacency_matrix_week_filename, temporal_matrix_week_filename,
                           adjacency_matrix_weekend_filename, temporal_matrix_weekend_filename,
                           distance_matrix_filename, distance_matrix_weekday_filename, distance_magrix_weekend_filename,
                           duration_matrix_filename, duration_matrix_weekday_filename, duration_matrix_weekend_filename)

        # normal matrices
        adjacency_df, temporal_df, distance_df, duration_df = self.poi_categorization_domain.\
            read_matrix(adjacency_matrix_filename, temporal_matrix_filename, distance_matrix_filename, duration_matrix_filename)
        print("arquivos: \n", adjacency_matrix_filename)
        print(temporal_matrix_filename)

        # week matrices
        adjacency_week_df, temporal_week_df, distance_week_df, duration_week_df = self.poi_categorization_domain. \
            read_matrix(adjacency_matrix_week_filename, temporal_matrix_week_filename, distance_matrix_weekday_filename, duration_matrix_weekday_filename)
        # weekend matrices
        adjacency_weekend_df, temporal_weekend_df, distance_weekend_df, duration_weekend_df = self.poi_categorization_domain. \
            read_matrix(adjacency_matrix_weekend_filename, temporal_matrix_weekend_filename,
                        distance_magrix_weekend_filename, duration_matrix_weekday_filename)
        print("Verificação de matrizes")
        self.matrices_verification(adjacency_df, temporal_df, adjacency_week_df, temporal_week_df,
                                   adjacency_weekend_df, temporal_weekend_df, distance_df, distance_week_df, distance_weekend_df, duration_df, duration_week_df, duration_weekend_df)


        inputs = {'all_week': {'adjacency': adjacency_df, 'temporal': temporal_df, 'distance': distance_df, 'duration': duration_df},
                  'week': {'adjacency': adjacency_week_df, 'temporal': temporal_week_df, 'distance': distance_week_df, 'duration': duration_week_df},
                  'weekend': {'adjacency': adjacency_weekend_df, 'temporal': temporal_weekend_df, 'distance': distance_weekend_df, 'duration': duration_weekend_df}}

        print("Preprocessing")
        users_categories, adjacency_df, temporal_df, distance_df, duration_df, adjacency_week_df, temporal_week_df,\
        distance_week_df, duration_week_df, adjacency_weekend_df, temporal_weekend_df, \
        distance_weekend_df, duration_weekend_df = self.poi_categorization_domain.adjacency_preprocessing(inputs,
                                    max_size_matrices,
                                    max_size_paths,
                                    True,
                                    True,
                                    7)

        self.matrices_verification(adjacency_df, temporal_df, adjacency_week_df, temporal_week_df,
                              adjacency_weekend_df, temporal_weekend_df, distance_df, distance_week_df,
                                   distance_weekend_df, duration_df, duration_week_df, duration_weekend_df)



        inputs = {'all_week': {'adjacency': adjacency_df, 'temporal': temporal_df,
                               'categories': users_categories, 'distance': distance_df, 'duration': duration_df},
                  'week': {'adjacency': adjacency_week_df, 'temporal': temporal_week_df,
                           'categories': users_categories, 'distance': distance_week_df, 'duration': duration_week_df},
                  'weekend': {'adjacency': adjacency_weekend_df, 'temporal': temporal_weekend_df,
                              'categories': users_categories, 'distance': distance_weekend_df,
                              'duration': duration_weekend_df}}

        usuarios = len(adjacency_df)

        folds, class_weight = self.poi_categorization_domain.\
            k_fold_split_train_test(max_size_matrices,
                                    inputs,
                                    n_splits,
                                    'all_week')

        folds_week, class_weight_week = self.poi_categorization_domain. \
            k_fold_split_train_test(max_size_matrices,
                                    inputs,
                                    n_splits,
                                    'week')

        folds_weekend, class_weight_weekend = self.poi_categorization_domain. \
            k_fold_split_train_test(max_size_matrices,
                                    inputs,
                                    n_splits,
                                    'weekend')

        print("class weight: ", class_weight)
        inputs_folds = {'all_week': {'folds': folds, 'class_weight': class_weight},
                        'week': {'folds': folds_week, 'class_weight': class_weight_week},
                        'weekend': {'folds': folds_weekend, 'class_weight': class_weight_weekend}}

        print("Treino")
        folds_histories, base_report, model = self.poi_categorization_domain.\
            k_fold_with_replication_train_and_evaluate_model(inputs_folds,
                                                             n_replications,
                                                             max_size_matrices,
                                                             max_size_paths,
                                                             base_report,
                                                             epochs,
                                                             class_weight,
                                                             base,
                                                             country,
                                                             version,
                                                             output_dir)

        print("------------- Location ------------")
        print(base_report)
        base_report = self.poi_categorization_domain.preprocess_report(base_report, int_to_category)
        self.poi_categorization_loader.plot_history_metrics(folds_histories, base_report, output_dir)
        self.poi_categorization_loader.save_report_to_csv(output_dir, base_report, n_splits, n_replications, usuarios)
        self.poi_categorization_loader.save_model_and_weights(model, output_dir, n_splits, n_replications)
        print("Usuarios processados: ", usuarios)
        print("Tamanho máximo de matriz: ", max_size_matrices)
        print("Quantidade mínima de registros: ", max_size_paths)

    def files_verification(self, country, state, adjacency_matrix_filename, temporal_matrix_filename,
                           adjacency_matrix_week_filename, temporal_matrix_week_filename,
                           adjacency_matrix_weekend_filename, temporal_matrix_weekend_filename,
                           distance_matrix_filename, distance_matrix_weekday_filename, distance_magrix_weekend_filename,
                           duration_matrix_filename, duration_matrix_weekday_filename, duration_matrix_weekend_filename):

        if country not in adjacency_matrix_filename or country not in temporal_matrix_filename \
                or country not in adjacency_matrix_week_filename or country not in adjacency_matrix_weekend_filename or country not in \
                distance_matrix_filename \
                or country not in distance_matrix_weekday_filename or country not in \
                distance_magrix_weekend_filename or country not in duration_matrix_filename or country not in \
                duration_matrix_weekday_filename or country not in duration_matrix_weekday_filename:

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

        if 'week' not in adjacency_matrix_week_filename or 'week' not in temporal_matrix_week_filename or 'week' not in distance_matrix_weekday_filename or 'week' not in duration_matrix_weekday_filename:
            print("matrizes diferentes de week")
            print(adjacency_matrix_week_filename)
            print(temporal_matrix_week_filename)
            print(distance_matrix_weekday_filename)
            print(duration_matrix_weekday_filename)
            raise

        if 'weekend' not in adjacency_matrix_weekend_filename or 'weekend' not in temporal_matrix_weekend_filename or 'weekend' not in distance_magrix_weekend_filename or 'weekend' not in duration_matrix_weekend_filename:
            print("matrizes diferentes weekend")
            print(adjacency_matrix_weekend_filename)
            print(temporal_matrix_weekend_filename)
            raise

    def matrices_verification(self, adjacency_df, temporal_df, adjacency_week_df, temporal_week_df,
                              adjacency_weekend_df, temporal_weekend_df,  distance_df, distance_weekday_df, distance_weekend_df, duration_df, duration_weekday_df, duration_weekend_df):

        if not(len(adjacency_df) == len(temporal_df) == len(adjacency_week_df) == len(temporal_week_df) == len(adjacency_weekend_df) == len(temporal_weekend_df) == len(distance_df) == len(distance_weekday_df) == len(distance_weekend_df) == len(duration_df) == len(duration_weekday_df) == len(duration_weekend_df)):
            print("Matrizes com tamanhos diferentes")
            raise
        else:
            print("Quantidade inicial de usuários: ", len(adjacency_df))