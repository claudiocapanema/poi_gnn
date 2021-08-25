from configuration.poi_categorization_configuration import PoICategorizationConfiguration
from configuration.poi_categorization_baselines_configuration import PoICategorizationBaselinesConfiguration
from configuration.poi_categorization_performance_graphics_configuration import PoiCategorizationPerformanceGraphicsConfiguration
from domain.poi_categorization_performance_graphics_domain import PoiCategorizationPerformanceGraphicsDomain
from loader.poi_categorization_performance_graphics_loader import PoiCategorizationPerformanceGraphicsLoader
#from configuration.poi_categorization_sequential_baselines_configuration import PoiCategorizationSequentialBaselinesConfiguration
from foundation.configuration.input import Input

class PoiCategorizationPerformanceGraphicsJob:

    def __init__(self):
        self.poi_categorization_configuration = PoICategorizationConfiguration()
        self.poi_categorization_baselines_configuration = PoICategorizationBaselinesConfiguration()
        self.poi_categorization_performance_graphics_domain = PoiCategorizationPerformanceGraphicsDomain()
        self.poi_categorization_performance_graphics_loader = PoiCategorizationPerformanceGraphicsLoader()

    def start(self):
        dataset_name = Input.get_instance().inputs['dataset_name']
        categories_type = Input.get_instance().inputs['categories_type']
        folds = Input.get_instance().inputs['folds']
        replications = Input.get_instance().inputs['replications']
        graph_type = Input.get_instance().inputs['graph_type']
        country = Input.get_instance().inputs['country']
        version = Input.get_instance().inputs['version']

        sequential_model_names = PoiCategorizationPerformanceGraphicsConfiguration.SEQUENTIAL_POI_RECOMMENDATION_BASELINES_MODELS_NAMES.get_value()

        output_base_dir_poi_categorization = self.poi_categorization_configuration.OUTPUT_DIR[1]
        model_names = PoiCategorizationPerformanceGraphicsConfiguration.MODELS_NAMES.get_value()
        folds_replications = PoiCategorizationPerformanceGraphicsConfiguration.FOLDS_REPLICATIONS.get_value()[folds][replications]
        folds_replications_filename = PoiCategorizationPerformanceGraphicsConfiguration.FOLDS_REPLICATIONS_FILENAME.get_value()[folds][replications]
        graph_type_dir = self.poi_categorization_configuration.GRAPH_TYPE[1][graph_type]
        graph_type_directed_dir = self.poi_categorization_configuration.GRAPH_TYPE[1]['directed']
        country_dir = self.poi_categorization_baselines_configuration.COUNTRY[1][country]
        version_dir = self.poi_categorization_configuration.VERSION[1][version]
        base_dir = PoiCategorizationPerformanceGraphicsConfiguration.BASE_DIR.get_value()[dataset_name][graph_type][country][version]
        osm_categories_to_int = self.poi_categorization_baselines_configuration. \
            CATEGORIES[1][dataset_name][categories_type]

        output_dirs = []

        # getting the metrics.csv directories of the baselines
        new_models_names = []
        for model_name in model_names:
            if model_name == 'gae' or model_name == 'gcn' or model_name not in ['gcn', 'arma']:
                continue
            output_base_dir = self.poi_categorization_baselines_configuration.OUTPUT_BASE_DIR[1]
            dataset_type_dir = self.poi_categorization_baselines_configuration.DATASET_TYPE[1][dataset_name]
            category_type_dir = self.poi_categorization_baselines_configuration.CATEGORY_TYPE[1][categories_type]
            model_name_dir = self.poi_categorization_baselines_configuration.MODEL_NAME[1][model_name]
            if model_name == "gpr":
                output_dir = self.poi_categorization_baselines_configuration. \
                    output_dir(output_base_dir=output_base_dir, graph_type=graph_type_directed_dir,
                               dataset_type=dataset_type_dir, category_type=category_type_dir, country=country_dir,
                               model_name=model_name_dir)
            else:
                output_dir = self.poi_categorization_baselines_configuration. \
                    output_dir(output_base_dir=output_base_dir, graph_type=graph_type_dir,
                               dataset_type=dataset_type_dir, category_type=category_type_dir, country=country_dir,
                               model_name=model_name_dir)

            output_dirs.append(output_dir+folds_replications)
            new_models_names.append(model_name)
        # getting the metrics.csv directories of POI recommendation baselines
        # for model_name in sequential_model_names:
        #     model_names.append(model_name)
        #     output_base_dir = PoiCategorizationSequentialBaselinesConfiguration.OUTPUT_BASE_DIR.get_value()
        #     dataset_type_dir = PoiCategorizationSequentialBaselinesConfiguration.DATASET_TYPE.get_value()[dataset_name]
        #     category_type_dir = PoiCategorizationSequentialBaselinesConfiguration.CATEGORY_TYPE.get_value()[categories_type]
        #     model_name_dir = PoiCategorizationSequentialBaselinesConfiguration.MODEL_NAME.get_value()[model_name]
        #     output_dir = output_base_dir + dataset_type_dir + category_type_dir + model_name_dir
        #
        #     output_dirs.append(output_dir + folds_replications)

        # getting the metrics.csv directory of the hmrm
        # if country != 'JP' and country != 'US':
        #     model_name = 'hmrm'
        #     new_models_names.append(model_name)
        #     output_dirs.append("/home/claudio/Documentos/pycharm_projects/poi_detection/output/matrix_factorization_baseline/global_foursquare/BR/8_categories/5_folds/all_year_long/metrics_8_categories_no_balanced_weight.csv")

        # getting the metrics.csv directory of the poi-gnn
        # model_name = 'POI-GNN'
        # new_models_names.append(model_name)
        # output_base_dir = self.poi_categorization_configuration.OUTPUT_DIR[1]
        # dataset_type_dir = self.poi_categorization_configuration.DATASET_TYPE[1][dataset_name]
        # category_type_dir = self.poi_categorization_configuration.CATEGORY_TYPE[1][categories_type]
        # output_dir = self.poi_categorization_baselines_configuration. \
        #     output_dir(output_base_dir=output_base_dir, graph_type=graph_type_dir, dataset_type=dataset_type_dir,
        #                category_type=category_type_dir, country=country_dir, version=version_dir)
        # output_dirs.append(output_dir+folds_replications)

        print("ler", output_dirs)
        metrics = self.poi_categorization_performance_graphics_domain.\
            read_metrics(output_dirs, new_models_names, folds_replications)
        print("metrica", metrics)
        metrics = metrics[['0_fscore', '1_fscore', '2_fscore', '3_fscore', '4_fscore', '5_fscore', '6_fscore', '7_fscore', 'accuracy', 'macro_avg_fscore', 'weighted_avg_fscore', 'Method']]
        self.poi_categorization_performance_graphics_domain.\
            performance_graphics(metrics, osm_categories_to_int, base_dir, folds_replications_filename)





