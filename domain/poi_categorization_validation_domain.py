import numpy as np
import pandas as pd

from model.confusion_matrix import ConfusionMatrix
from configuration.categorized_points_of_interest_validation_configuration import CategorizedPointsOfInterestValidationConfiguration
from domain.pois_around_domain import PoisAroundDomain

class CategorizationValidation:

    def __init__(self):
        self.pois_around = PoisAroundDomain(CategorizedPointsOfInterestValidationConfiguration.METERS_AROUND)
        ##iniciar todas as categorias aqui


    def start(self, ground_truth, categorized):
        print("------ validation -----")

        ##Como vão estar os datasets?
        ##users_checkins = grount_truth["placeid"] ....
        ##ids = users_checkins["userids"].unique()

        for i in ids:
            
            ##Filtrar pontos de acordo com o dataset
            ##categorized_points = categorized["category"].tolist()

            keys, values = self.pois_around.generate_pois_around_for_unique_point(latitude_ground_truth, longitude_ground_truth) ##osm pega todas as keys e values proximas do ponto

            ##mapear essas keys e values
            ##category_maped = categorias_mapeadas

            for point in user_categorized_points: ##cada lugar vai ter um id?
                
                if(point in categorias_mapeada):
                    self._add_tp(categoria) 
                else:
                    self._add_fp(categoria)
                
                ##todas as categorias mapeadas vão ser fn?
                ##Fn e tn são o que?
    


    def _add_tp(self, poi_type):
        if poi_type == categorias: ##adicionar cnodição para todas as categorias 
            self.categoria_confusion_matrix.add_tp()


    def _add_fp(self, poi_type):
        if poi_type == categorias: ##adicionar cnodição todas as categorias
            self.categoria_confusion_matrix.add_fp()
        

    def _add_fn(self, poi_type):
        if poi_type == categorias: ##adicionar cnodição todas as categorias
            self.categoria_confusion_matrix.add_fn()

    def _add_tn(self, poi_type):
        if poi_type == categorias: ##adicionar cnodição todas as categorias
            self.categoria_confusion_matrix.add_tn()

            





        
