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
        self.poi_categorization_transfer_learning_domain = PoiCategorizationTransferLearningDomain(Input.get_instance().inputs['dataset_name'])
        self.poi_categorization_loader = PoiCategorizationLoader()
        self.poi_categorization_configuration = PoICategorizationConfiguration()

    def start(self):
        adjacency_matrix_filename = Input.get_instance().inputs['adjacency_matrix_filename']
        graph_type = Input.get_instance().inputs['graph_type']
        feature_matrix_filename = Input.get_instance().inputs['temporal_matrix_filename']
        dataset_name = Input.get_instance().inputs['dataset_name']
        categories_type = Input.get_instance().inputs['categories_type']
        ground_truth = Input.get_instance().inputs['ground_truth']
        print("Dataset: ", Input.get_instance().inputs['dataset_name'])

        # get list of valid categories for the given dataset
        categories_to_int_osm = self.poi_categorization_configuration.\
            DATASET_CATEGORIES_TO_INT_OSM_CATEGORIES[1][dataset_name][categories_type]
        max_size_matrices = self.poi_categorization_configuration.MAX_SIZE_MATRICES[1]
        train_size = self.poi_categorization_configuration.TRAIN_SIZE[1]
        n_splits = self.poi_categorization_configuration.N_SPLITS[1]
        n_replications = self.poi_categorization_configuration.N_REPLICATIONS[1]

        output_base_dir = self.poi_categorization_configuration.OUTPUT_DIR[1]
        dataset_type_dir = self.poi_categorization_configuration.DATASET_TYPE[1][dataset_name]
        category_type_dir = self.poi_categorization_configuration.CATEGORY_TYPE[1][categories_type]
        graph_type_dir = self.poi_categorization_configuration.GRAPH_TYPE[1][graph_type]
        output_dir = self.poi_categorization_configuration.output_dir(output_base_dir, graph_type_dir, dataset_type_dir, category_type_dir)

        base_report = self.poi_categorization_configuration.REPORT_MODEL[1][categories_type]

        filename = output_dir + str(n_splits) + "_folds/" + str(n_replications) + "_replications/"
        base_model = self.poi_categorization_transfer_learning_domain.read_model(filename)

        # raw gps matrices
        adjacency_df, feature_df = self.poi_categorization_transfer_learning_domain.\
            read_matrix(adjacency_matrix_filename, feature_matrix_filename)

        users_metrics = self.poi_categorization_transfer_learning_domain.read_users_metrics(user_metrics_filename)
        users_metrics_ids = users_metrics['user_id'].unique().tolist()

        adjacency_df, users_categories, feature_df, remove_users_ids = self.poi_categorization_transfer_learning_domain.\
            adjacency_preprocessing(adjacency_df, feature_df, users_metrics_ids, categories_to_int_osm,
                                    max_size_matrices, categories_type)

        users_metrics = users_metrics.query("user_id not in " + str(remove_users_ids))

        folds, class_weight = self.poi_categorization_transfer_learning_domain.\
            k_fold_split_train_test(adjacency_df,
                                    users_categories,
                                    feature_df,
                                    train_size,
                                    n_splits,
                                    users_metrics)

        folds_histories, base_report, model = self.poi_categorization_transfer_learning_domain.\
            k_fold_with_replication_train_and_evaluate_model(base_model,
                                                             folds,
                                                             n_replications,
                                                             class_weight,
                                                             categories_to_int_osm,
                                                             max_size_matrices,
                                                             base_report)

        print("------------- Location ------------")
        print(base_report)
        self.poi_categorization_loader.plot_history_metrics(folds_histories, base_report, output_dir)
        self.poi_categorization_loader.save_report_to_csv(output_dir, base_report, n_splits, n_replications)
        self.poi_categorization_loader.save_model_and_weights(model, output_dir, n_splits, n_replications)

        classes = len(pd.Series(list(categories_to_int_osm.values())).unique().tolist())
        self.poi_categorization_transfer_learning_domain.transfer_learning(classes, max_size_matrices, base_model)
        print(base_model)