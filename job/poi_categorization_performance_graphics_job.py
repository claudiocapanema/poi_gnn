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
        state = Input.get_instance().inputs['state']
        version = Input.get_instance().inputs['version']
        base = Input.get_instance().inputs['base']

        if len(base) > 0:
            base = base + "/"

        output_base_dir_poi_categorization = self.poi_categorization_configuration.OUTPUT_DIR[1]
        model_names = PoiCategorizationPerformanceGraphicsConfiguration.MODELS_NAMES.get_value()
        folds_replications = PoiCategorizationPerformanceGraphicsConfiguration.FOLDS_REPLICATIONS.get_value()[folds][replications]
        folds_replications_filename = PoiCategorizationPerformanceGraphicsConfiguration.FOLDS_REPLICATIONS_FILENAME.get_value()[folds][replications]
        graph_type_dir = self.poi_categorization_configuration.GRAPH_TYPE[1][graph_type]
        graph_type_directed_dir = self.poi_categorization_configuration.GRAPH_TYPE[1]['directed']
        country_dir = self.poi_categorization_baselines_configuration.COUNTRY[1][country]
        state_dir = self.poi_categorization_baselines_configuration.STATE[1][state]
        version_dir = self.poi_categorization_configuration.VERSION[1][version]
        base_dir = PoiCategorizationPerformanceGraphicsConfiguration.BASE_DIR.get_value()[dataset_name][graph_type][country][state][version]
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
                    output_dir(output_base_dir=output_base_dir, base="", graph_type=graph_type_directed_dir,
                               dataset_type=dataset_type_dir, category_type=category_type_dir, country=country_dir, state_dir=state_dir,
                               model_name=model_name_dir)
            else:
                output_dir = self.poi_categorization_baselines_configuration. \
                    output_dir(output_base_dir=output_base_dir, base="", graph_type=graph_type_dir,
                               dataset_type=dataset_type_dir, category_type=category_type_dir, country=country_dir, state_dir=state_dir,
                               model_name=model_name_dir)

            output_dirs.append(output_dir+folds_replications)
            new_models_names.append(model_name)

        # getting the metrics.csv directory of the hmrm
        if (country == 'US' and dataset_name == "gowalla" and state == "TX"):
            model_name = 'hmrm'
            new_models_names.append(model_name)
            output_dirs.append("/home/claudio/Documentos/pycharm_projects/poi_gnn/output/poi_categorization_baselines_job/not_directed/gowalla/US/TX/7_categories/hmrm/5_folds/1_replications/")

        # getting the metrics.csv directory of the poi-gnn
        model_name = 'POI-GNN'
        new_models_names.append(model_name)
        output_base_dir = self.poi_categorization_configuration.OUTPUT_DIR[1]
        dataset_type_dir = self.poi_categorization_configuration.DATASET_TYPE[1][dataset_name]
        category_type_dir = self.poi_categorization_configuration.CATEGORY_TYPE[1][categories_type]
        output_dir = self.poi_categorization_baselines_configuration. \
            output_dir(output_base_dir=output_base_dir, base=base, graph_type=graph_type_dir, dataset_type=dataset_type_dir,
                       category_type=category_type_dir, country=country_dir, state_dir=state_dir, version=version_dir)
        output_dirs.append(output_dir+folds_replications)

        print("ler", output_dirs)
        # metrics = self.poi_categorization_performance_graphics_domain.\
        #     read_metrics(output_dirs, new_models_names, folds_replications)
        # print("metrica", metrics)
        self.poi_categorization_performance_graphics_domain.\
            performance_graphics(output_dirs, new_models_names, osm_categories_to_int, base_dir, dataset_name)





