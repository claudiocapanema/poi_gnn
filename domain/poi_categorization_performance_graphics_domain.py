import pandas as pd

from extractor.file_extractor import FileExtractor
from loader.poi_categorization_performance_graphics_loader import PoiCategorizationPerformanceGraphicsLoader

class PoiCategorizationPerformanceGraphicsDomain:

    def __init__(self):
        self.file_extractor = FileExtractor()
        self.poi_categorization_performance_graphics_loader = PoiCategorizationPerformanceGraphicsLoader()

    def read_metrics(self, files_names, method_name, folds_replications):

        df_metrics = None
        column = 'Method'
        for i in range(len(files_names)):
            df = self.file_extractor.read_csv(files_names[i])
            df[column] = pd.Series([method_name[i]]*df.shape[0])
            if df_metrics is not None:
                df_metrics = pd.concat([df_metrics, df], ignore_index=True)
            else:
                df_metrics = df

        return df_metrics

    def performance_graphics(self, metrics, osm_categories_to_int, base_dir, folds_replications_filename):
        self.poi_categorization_performance_graphics_loader.plot_metrics(metrics, osm_categories_to_int, base_dir, folds_replications_filename)
