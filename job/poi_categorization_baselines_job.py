
from domain.user_step_domain import UserStepDomain
from foundation.abs_classes.job import Job
from domain.poi_categorization_domain import PoiCategorizationDomain
from extractor.file_extractor import FileExtractor
from foundation.configuration.input import Input
from foundation.util.general_utils import join_df
from configuration.poi_categorization_baselines_configuration import PoICategorizationBaselinesConfiguration
from configuration.poi_categorization_configuration import PoICategorizationConfiguration
from domain.poi_categorization_baselines_domain import PoiCategorizationBaselinesDomain
from loader.poi_categorization_loader import PoiCategorizationLoader

class PoiCategorizationBaselinesJob:

    def __init__(self):
        self.user_step_domain = UserStepDomain()
        self.file_extractor = FileExtractor()
        self.poi_categorization_baselines_domain = PoiCategorizationBaselinesDomain(Input.get_instance().inputs['dataset_name'])
        self.poi_categorization_loader = PoiCategorizationLoader()
        self.poi_categorization_baselines_configuration = PoICategorizationBaselinesConfiguration()
        self.poi_categorization_configuration = PoICategorizationConfiguration()
        self.poi_categorization_domain = PoiCategorizationDomain(Input.get_instance().inputs['dataset_name'])

    def start(self):
        base_dir = Input.get_instance().inputs['base_dir']
        adjacency_matrix_filename = Input.get_instance().inputs['adjacency_matrix_filename']
        graph_type = Input.get_instance().inputs['graph_type']
        temporal_matrix_filename = Input.get_instance().inputs['temporal_matrix_filename']
        dataset_name = Input.get_instance().inputs['dataset_name']
        categories_type = Input.get_instance().inputs['categories_type']
        model_name = Input.get_instance().inputs['baseline']
        country = Input.get_instance().inputs['country']
        state = Input.get_instance().inputs['state']
        print("Dataset: ", Input.get_instance().inputs['dataset_name'])

        if country == 'Brazil':
            country_aux = 'BR'
        elif country == 'US':
            country_aux = 'US'

        self.files_verificaiton(country_aux, adjacency_matrix_filename, temporal_matrix_filename)

        max_size_matrices = self.poi_categorization_configuration.MAX_SIZE_MATRICES[1]
        max_size_sequence = self.poi_categorization_configuration.MINIMUM_RECORDS[1]
        n_splits = self.poi_categorization_configuration.N_SPLITS[1]
        n_replications = self.poi_categorization_configuration.N_REPLICATIONS[1]

        parameters = self.poi_categorization_baselines_configuration.PARAMETERS[1][country][model_name]

        output_base_dir = self.poi_categorization_baselines_configuration.OUTPUT_BASE_DIR[1]
        dataset_type_dir = self.poi_categorization_baselines_configuration.DATASET_TYPE[1][dataset_name]
        country_dir = self.poi_categorization_baselines_configuration.COUNTRY[1][country]
        state_dir = self.poi_categorization_baselines_configuration.STATE[1][state]
        category_type_dir = self.poi_categorization_baselines_configuration.CATEGORY_TYPE[1][categories_type]
        int_to_category = self.poi_categorization_configuration.INT_TO_CATEGORIES[1][dataset_name]
        model_name_dir = self.poi_categorization_baselines_configuration.MODEL_NAME[1][model_name]
        graph_type_dir = self.poi_categorization_baselines_configuration.GRAPH_TYPE[1][graph_type]
        units = self.poi_categorization_baselines_configuration.UNITS[1][dataset_name][country][model_name]
        output_dir = self.poi_categorization_baselines_configuration.\
            output_dir(output_base_dir=output_base_dir, graph_type=graph_type_dir,
                       dataset_type=dataset_type_dir, country=country_dir, state_dir=state_dir, category_type=category_type_dir,
                       model_name=model_name_dir)

        base_report = self.poi_categorization_baselines_configuration.REPORT_MODEL[1][categories_type]

        base_dir = base_dir + graph_type + "/" + country_dir + state + "/"
        adjacency_matrix_filename = base_dir + adjacency_matrix_filename
        temporal_matrix_filename = base_dir + temporal_matrix_filename
        adjacency_df, temporal_df, aux1, aux2 = self.poi_categorization_baselines_domain. \
            read_matrix(adjacency_matrix_filename, temporal_matrix_filename)
        print("arquivos: \n", adjacency_matrix_filename)
        print(temporal_matrix_filename)

        self.matrices_verification(adjacency_df, temporal_df)

        inputs = {'all_week': {'adjacency': adjacency_df, 'temporal': temporal_df}}
        adjacency_df, users_categories, temporal_df = self.poi_categorization_baselines_domain. \
            adjacency_preprocessing(inputs,
                                    max_size_matrices,
                                    max_size_sequence,
                                    False,
                                    False,
                                    7,
                                    model_name)

        usuarios = adjacency_df.shape

        inputs = {'all_week': {'adjacency': adjacency_df, 'temporal': temporal_df, 'categories': users_categories}}
        folds, class_weight = self.poi_categorization_baselines_domain. \
            k_fold_split_train_test(max_size_matrices,
                                    inputs,
                                    n_splits,
                                    'all_week',
                                    model_name=model_name)

        folds_histories, base_report = self.poi_categorization_baselines_domain.\
            k_fold_with_replication_train_and_evaluate_baselines_model(folds,
                                                                        n_replications,
                                                                        class_weight,
                                                                        max_size_matrices,
                                                                        base_report,
                                                                        parameters,
                                                                        model_name,
                                                                        units,
                                                                        country)

        print("------------- Location ------------")
        print(base_report)
        base_report = self.poi_categorization_domain.preprocess_report(base_report, int_to_category)
        self.poi_categorization_loader.plot_history_metrics(folds_histories, base_report, output_dir)
        self.poi_categorization_loader.save_report_to_csv(output_dir, base_report, n_splits, n_replications, usuarios)

    def matrices_verification(self, adjacency_df, temporal_df):

        if not(len(adjacency_df) == len(temporal_df)):
            print("Matrizes com tamanhos diferentes")
            raise
        else:
            print("Quantidade inicial de usuários: ", len(adjacency_df))

    def files_verificaiton(self, country, adjacency_matrix_filename, temporal):

        if country == 'Brazil':
            country = 'BR'

        if country not in adjacency_matrix_filename or country not in temporal:
            print("País diferente")
            raise