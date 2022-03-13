
from domain.user_step_domain import UserStepDomain
from foundation.abs_classes.job import Job
from domain.poi_categorization_domain import PoiCategorizationDomain
from extractor.file_extractor import FileExtractor
from foundation.configuration.input import Input
from foundation.util.general_utils import join_df
from configuration.poi_categorization_baselines_configuration import PoICategorizationBaselinesConfiguration
from domain.poi_categorization_baseline_gpr_domain import PoiCategorizationBaselineGPRDomain
from loader.poi_categorization_gpr_loader import PoiCategorizationGPRLoader

class PoiCategorizationBaselineGPRJob:

    def __init__(self):
        self.user_step_domain = UserStepDomain()
        self.file_extractor = FileExtractor()
        self.poi_categorization_baseline_gpr_domain = PoiCategorizationBaselineGPRDomain(Input.get_instance().inputs['dataset_name'])
        self.poi_categorization_gpr_loader = PoiCategorizationGPRLoader()
        self.poi_categorization_baselines_configuration = PoICategorizationBaselinesConfiguration()

    def start(self):
        adjacency_matrix_filename = Input.get_instance().inputs['adjacency_matrix_filename']
        graph_type = Input.get_instance().inputs['graph_type']
        feature_matrix_filename = Input.get_instance().inputs['feature_matrix_filename']
        user_poi_vector_filename = Input.get_instance().inputs['user_poi_vector_filename']
        dataset_name = Input.get_instance().inputs['dataset_name']
        categories_type = Input.get_instance().inputs['categories_type']
        country = Input.get_instance().inputs['country']
        model_name = "gpr"
        print("Dataset: ", Input.get_instance().inputs['dataset_name'])

        if 'United States' not in adjacency_matrix_filename or 'United States' not in feature_matrix_filename:
            print("matrizes diferentes do país")
            raise

        # get list of valid categories for the given dataset
        # categories_to_int_osm = self.poi_categorization_baselines_configuration. \
        #     DATASET_CATEGORIES_TO_INT_OSM_CATEGORIES[1][dataset_name][categories_type]
        max_size_matrices = self.poi_categorization_baselines_configuration.MAX_SIZE_MATRICES[1]
        n_splits = self.poi_categorization_baselines_configuration.N_SPLITS[1]
        n_replications = self.poi_categorization_baselines_configuration.N_REPLICATIONS[1]

        parameters = self.poi_categorization_baselines_configuration.PARAMETERS[1][country][model_name]

        output_base_dir = self.poi_categorization_baselines_configuration.OUTPUT_BASE_DIR[1]
        dataset_type_dir = self.poi_categorization_baselines_configuration.DATASET_TYPE[1][dataset_name]
        category_type_dir = self.poi_categorization_baselines_configuration.CATEGORY_TYPE[1][categories_type]
        country_dir = self.poi_categorization_baselines_configuration.COUNTRY[1][country]
        model_name_dir = self.poi_categorization_baselines_configuration.MODEL_NAME[1][model_name]
        graph_type_dir = self.poi_categorization_baselines_configuration.GRAPH_TYPE[1][graph_type]
        output_dir = output_base_dir + graph_type_dir + dataset_type_dir + category_type_dir + country_dir + model_name_dir

        base_report = self.poi_categorization_baselines_configuration.REPORT_MODEL[1][categories_type]
        augmentation_categories = self.poi_categorization_baselines_configuration.AUGMENTATION_CATEGORIES[1]

        adjacency_df, distance_df, user_poi_vector_df = self.poi_categorization_baseline_gpr_domain. \
            read_matrix(adjacency_matrix_filename, feature_matrix_filename, user_poi_vector_filename)

        adjacency_df, users_categories, distance_df, user_poi_df_df, remove_users_ids = self.poi_categorization_baseline_gpr_domain. \
            adjacency_preprocessing(adjacency_df, distance_df, user_poi_vector_df, max_size_matrices, categories_type, model_name)

        inputs = {'adjacency': adjacency_df, 'distance': distance_df, 'user_poi': user_poi_df_df, 'categories': users_categories}

        folds = self.poi_categorization_baseline_gpr_domain. \
            k_fold_split_train_test_gpr(5,
                                        inputs,
                                    n_splits)

        class_weight = {}

        folds_histories, base_report = self.poi_categorization_baseline_gpr_domain.\
            k_fold_with_replication_train_and_evaluate_baselines_model(folds,
                                                             n_replications,
                                                             class_weight,
                                                             max_size_matrices,
                                                             base_report,
                                                            parameters,
                                                            country)

        print("------------- Location ------------")
        print(base_report)
        self.poi_categorization_gpr_loader.save_report_to_csv(output_dir, base_report, n_splits, n_replications)
        self.poi_categorization_gpr_loader.plot_history_metrics(folds_histories, base_report, output_dir)
