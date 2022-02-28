from .base_poi_categorization_configuration import BasePoiCategorizationConfiguration

class PoICategorizationConfiguration(BasePoiCategorizationConfiguration):

    def __init__(self):
        super(PoICategorizationConfiguration, self).__init__()

        self.OUTPUT_DIR = ("output_dir", "output/poi_categorization_job/", False, "output directory for the poi_categorization_job")

        self.EPOCHS = ("epochs", {'BR': 18, 'US': 20, 'Brazil': 12, 'United States': 12})

        self.VERSION = ("version", {"normal": ""})