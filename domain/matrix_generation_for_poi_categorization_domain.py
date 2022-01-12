import math
import pandas as pd
import numpy as np
import time
import statistics as st

from foundation.util.geospatial_utils import points_distance
from loader.matrix_generation_for_poi_categorization_loarder import MatrixGenerationForPoiCategorizationLoader
from extractor.file_extractor import FileExtractor


class MatrixGenerationForPoiCategorizationDomain:


    def __init__(self, dataset_name):
        self.matrix_generation_for_poi_categorization_loader = MatrixGenerationForPoiCategorizationLoader()
        self.file_extractor = FileExtractor()
        self.dataset_name = dataset_name
        self.distance_sigma = 10
        self.duration_sigma = 10
        self.max_events = 200

    def filter_user(self, user_checkin,
                    userid_column,
                   userid,
                   datetime_column,
                   category_column):

        user_checkin = user_checkin.sort_values(by=[datetime_column]).head(self.max_events)
        categories = user_checkin[category_column].tolist()
        # converter os nomes das categorias para inteiro
        # if osm_category_column is not None:
        #     # pre-processar raw gps
        #     categories = self.categories_list_preproccessing(categories, categories_to_int_osm)
        #
        # else:
        #     categories = self.categories_preproccessing(categories, categories_to_int_osm)

        if len(user_checkin[category_column].unique().tolist()) < 7:
            return pd.DataFrame({'tipo': ['nan'], userid_column: [userid]})
        else:
            return pd.DataFrame({'tipo': ['bom'], userid_column: [userid]})

    def generate_user_matrices(self, user_checkin,
                               userid,
                               datetime_column,
                               locationid_column,
                               category_column,
                               latitude_column,
                               longitude_column,
                               osm_category_column,
                               differemt_venues, personal_features_matrix, hour48, directed, max_time_between_records):
        """
        :param user_checkin:
        :param datetime_column:
        :param locationid_column:
        :param category_column:
        :param osm_category_column:
        :param personal_features_matrix:
        :param hour48:
        :param directed:
        :return: adjacency, temporal, and path matrices
        """

        user_checkin = user_checkin.sort_values(by=[datetime_column]).head(self.max_events)
        latitude_list = user_checkin[latitude_column].tolist()
        longitude_list = user_checkin[longitude_column].tolist()

        # matrices initialization
        n_pois = len(user_checkin[locationid_column].unique().tolist())
        adjacency_matrix = [[0 for i in range(n_pois)] for j in range(n_pois)]
        adjacency_weekday_matrix = [[0 for i in range(n_pois)] for j in range(n_pois)]
        adjacency_weekend_matrix = [[0 for i in range(n_pois)] for j in range(n_pois)]
        temporal_weekday_matrix = [[0 for i in range(24)] for j in range(n_pois)]
        temporal_weekend_matrix = [[0 for i in range(24)] for j in range(n_pois)]
        distance_matrix = [[0 for i in range(n_pois)] for j in range(n_pois)]
        distance_weekday_matrix = [[0 for i in range(n_pois)] for j in range(n_pois)]
        distance_weekend_matrix = [[0 for i in range(n_pois)] for j in range(n_pois)]
        duration_matrix = [[[] for i in range(n_pois)] for j in range(n_pois)]
        duration_weekday_matrix = [[[] for i in range(n_pois)] for j in range(n_pois)]
        duration_weekend_matrix = [[[] for i in range(n_pois)] for j in range(n_pois)]
        if personal_features_matrix or not hour48:
            temporal_matrix = [[0 for i in range(24)] for j in range(n_pois)]
        else:
            temporal_matrix = [[0 for i in range(48)] for j in range(n_pois)]
        path_matrix = [[0 for i in range(len(user_checkin))] for j in range(n_pois)]
        path_weekday_list = []
        path_weekend_list = []
        path_weekday_count = 0
        path_weekend_count = 0
        categories_list = [-1 for i in range(n_pois)]

        datetimes = user_checkin[datetime_column].tolist()
        placeids = user_checkin[locationid_column].tolist()
        placeids_unique = user_checkin[locationid_column].unique().tolist()
        placeids_unique_to_int = {placeids_unique[i]: i for i in range(len(placeids_unique))}
        # converter os ids dos locais para inteiro
        placeids_int = [placeids_unique_to_int[placeids[i]] for i in range(len(placeids))]
        categories = user_checkin[category_column].tolist()
        # converter os nomes das categorias para inteiro
        # if osm_category_column is not None:
        #     # pre-processar raw gps
        #     categories = self.categories_list_preproccessing(categories, categories_to_int_osm)
        #
        # else:
        #     categories = self.categories_preproccessing(categories, categories_to_int_osm)

        if len(user_checkin[category_column].unique().tolist()) < 7:
            print("Usuário com poucas categorias diferentes visitadas")
            # return pd.DataFrame({'adjacency': ['vazio'], 'adjacency_weekday': ['vazio'],
            #                      'adjacency_weekend': ['vazio'], 'temporal': ['vazio'],
            #                      'distance': ['vazio'], 'duration': ['vazio'],
            #                      'temporal_weekday': ['vazio'],
            #                      'temporal_weekend': ['vazio'],
            #                      'path': ['vazio'], 'path_weekday': ['vazio'],
            #                      'path_weekend': ['vazio'],
            #                      'distance_weekday': ['vazio'],
            #                      'distance_weekend': ['vazio'],
            #                      'duration_weekday': ['vazio'],
            #                      'duration_weekend': ['vazio'],
            #                      'category': ['vazio']})

        # matrices initialization - setting the first visit
        if not personal_features_matrix:
            if hour48:
                if datetimes[0].weekday() < 5:
                    hour = datetimes[0].hour
                    temporal_matrix[placeids_int[0]][hour] += 1
                    temporal_weekday_matrix[placeids_int[0]][hour] += 1
                    path_weekday_list.append([placeids_int[0], path_weekday_count])
                    path_weekday_count+=1
                else:
                    hour = datetimes[0].hour + 24
                    temporal_matrix[placeids_int[0]][hour] += 1
                    temporal_weekend_matrix[placeids_int[0]][hour - 24] += 1
                    path_weekend_list.append([placeids_int[0], path_weekend_count])
                    path_weekend_count += 1
            else:
                hour = datetimes[0].hour
                temporal_matrix[placeids_int[0]][hour] += 1
        else:
            if datetimes[0].weekday() < 5:
                temporal_matrix[placeids_int[0]][math.floor(datetimes[0].hour / 2)] += 1
            else:
                temporal_matrix[placeids_int[0]][math.floor(datetimes[0].hour / 2) + 12] += 1
        path_matrix[placeids_int[0]][0] = 1
        categories_list[0] = categories[0]

        count = 0
        max_timedelta = pd.Timedelta(days=2020)
        if len(max_time_between_records) > 0:
            max_timedelta = pd.Timedelta(days=int(max_time_between_records))
        for j in range(1, len(datetimes)):
            anterior = j - 1
            atual = j
            local_anterior = placeids_int[anterior]
            local_atual = placeids_int[atual]
            # retirar eventos muito esparços
            if len(max_time_between_records) > 0:
                if (datetimes[atual] - datetimes[anterior]) > max_timedelta:
                    continue
            # retirar eventos consecutivos em um mesmo estabelecimento
            if differemt_venues:
                if placeids[anterior] == placeids[atual]:
                    continue

            lat_before = latitude_list[anterior]
            lng_before = longitude_list[anterior]
            lat_current = latitude_list[atual]
            lng_current = longitude_list[atual]
            if distance_matrix[local_anterior][local_atual] == 0 and distance_weekday_matrix[local_atual][local_atual] == 0 and distance_weekend_matrix[local_anterior][local_atual] == 0:
                distance = int(points_distance([lat_before, lng_before], [lat_current, lng_current]) / 1000)
                distance = self._distance_importance(distance)
            else:
                distance = max([distance_matrix[local_anterior][local_atual], distance_weekday_matrix[local_atual][local_atual], distance_weekend_matrix[local_anterior][local_atual]])

            datetime_before = datetimes[anterior]
            datetime_current = datetimes[atual]
            duration = int((datetime_current - datetime_before).total_seconds() / 3600)
            duration = self._duration_importance(duration)
            distance_matrix[local_anterior][local_atual] = distance
            distance_matrix[local_atual][local_anterior] = distance
            duration_matrix[local_anterior][local_atual].append(duration)
            if datetimes[atual].weekday() < 5:
                distance_weekday_matrix[local_anterior][local_atual] = distance
                distance_weekday_matrix[local_atual][local_anterior] = distance
                duration_weekday_matrix[local_anterior][local_atual].append(duration)
            else:
                distance_weekend_matrix[local_anterior][local_atual] = distance
                distance_weekend_matrix[local_atual][local_anterior] = distance
                duration_weekend_matrix[local_anterior][local_atual].append(duration)

            if directed:
                adjacency_matrix[placeids_int[anterior]][placeids_int[atual]] += 1
                if datetimes[atual].weekday() < 5:
                    adjacency_weekday_matrix[placeids_int[anterior]][placeids_int[atual]] += 1
                    path_weekday_list.append([placeids_int[atual], path_weekday_count])
                    path_weekday_count+=1
                else:
                    adjacency_weekend_matrix[placeids_int[anterior]][placeids_int[atual]] += 1
                    path_weekend_list.append([placeids_int[atual], path_weekend_count])
                    path_weekend_count+=1
            else:
                adjacency_matrix[placeids_int[anterior]][placeids_int[atual]] += 1
                adjacency_matrix[placeids_int[atual]][placeids_int[anterior]] += 1

                if datetimes[atual].weekday() < 5:
                    adjacency_weekday_matrix[placeids_int[anterior]][placeids_int[atual]] += 1
                    adjacency_weekday_matrix[placeids_int[atual]][placeids_int[anterior]] += 1
                    path_weekday_list.append([placeids_int[atual], path_weekday_count])
                    path_weekday_count += 1
                else:
                    adjacency_weekend_matrix[placeids_int[anterior]][placeids_int[atual]] += 1
                    adjacency_weekend_matrix[placeids_int[atual]][placeids_int[anterior]] += 1
                    path_weekend_list.append([placeids_int[atual], path_weekend_count])
                    path_weekend_count += 1

            if not personal_features_matrix:
                if hour48:
                    if datetimes[atual].weekday() < 5:
                        hour = datetimes[atual].hour
                        temporal_matrix[placeids_int[atual]][hour] += 1
                        temporal_weekday_matrix[placeids_int[atual]][hour] += 1
                    else:
                        hour = datetimes[atual].hour + 24
                        temporal_matrix[placeids_int[atual]][hour] += 1
                        temporal_weekend_matrix[placeids_int[atual]][hour - 24] += 1
                else:
                    hour = datetimes[atual].hour
                    temporal_matrix[placeids_int[atual]][hour] += 1
            else:
                if datetimes[atual].weekday() < 5:
                    temporal_matrix[placeids_int[atual]][math.floor(datetimes[atual].hour / 2)] += 1
                else:
                    temporal_matrix[placeids_int[atual]][math.floor(datetimes[atual].hour / 2) + 12] += 1

            path_matrix[placeids_int[atual]][j] = 1
            categories_list[placeids_int[atual]] = categories[atual]

        if osm_category_column is not None:
            # pre-processar raw gps
            adjacency_matrix, temporal_matrix, categories_list = self.remove_raw_gps_pois_that_dont_have_categories(
                categories_list, adjacency_matrix, temporal_matrix)
            if adjacency_matrix != []:
                count += 1

        else:
            adjacency_matrix, features_matrix, categories_list = self.remove_gps_pois_that_dont_have_categories(
                categories_list, adjacency_matrix, temporal_matrix)
            adjacency_weekday_matrix, features_weekday_matrix, categories_weekday_list = self.remove_gps_pois_that_dont_have_categories(
                categories_list, adjacency_weekday_matrix, temporal_weekday_matrix)
            adjacency_weekend_matrix, features_weekend_matrix, categories_list = self.remove_gps_pois_that_dont_have_categories(
                categories_list, adjacency_weekend_matrix, temporal_weekend_matrix)
            # if adjacency_matrix != []:
            #     count += 1

            if len(adjacency_matrix) <= 2 or len(temporal_matrix) <= 2 or len(categories_list) <= 2:
                #print("atencao anterior")
                # matrizes pequenas
                pass

                # new_ids.append(id_)
                # adj_matrices_column.append(str(adjacency_matrix))
                # feat_matrices_column.append(str(features_matrix))
                # sequence_matrices_column.append(str(sequence_matrix))
                # categories_column.append(categories_list)

        # create path matrices
        path_weekday_matrix = [[0 for i in range(len(path_weekday_list))] for j in range(n_pois)]
        path_weekend_matrix = [[0 for i in range(len(path_weekend_list))] for j in range(n_pois)]
        for i in range(len(path_weekday_list)):
            path_weekday_matrix[path_weekday_list[i][0]][path_weekday_list[i][1]] = 1

        for i in range(len(path_weekend_list)):
            path_weekend_matrix[path_weekend_list[i][0]][path_weekend_list[i][1]] = 1

        duration_matrix = self._summarize_categories_distance_matrix(duration_matrix)
        duration_weekday_matrix = self._summarize_categories_distance_matrix(duration_weekday_matrix)
        duration_weekend_matrix = self._summarize_categories_distance_matrix(duration_weekend_matrix)

        return pd.DataFrame({'adjacency': [adjacency_matrix], 'adjacency_weekday': [adjacency_weekday_matrix],
                             'adjacency_weekend': [adjacency_weekend_matrix], 'temporal': [temporal_matrix],
                             'distance': [distance_matrix], 'duration': [duration_matrix],
                             'temporal_weekday': [temporal_weekday_matrix], 'temporal_weekend': [temporal_weekend_matrix],
                             'path': [path_matrix], 'path_weekday': [path_weekday_matrix], 'path_weekend': [path_weekend_matrix],
                             'distance_weekday': [distance_weekday_matrix], 'distance_weekend': [distance_weekend_matrix],
                             'duration_weekday': [duration_weekday_matrix], 'duration_weekend': [duration_weekend_matrix],
                             'category': [categories_list]})

    def generate_pattern_matrices(self,
                                  users_checkin,
                                  adjacency_matrix_filename,
                                  adjacency_weekday_matrix_filename,
                                  adjacency_weekend_matrix_filename,
                                  temporal_matrix_filename,
                                  temporal_weekday_matrix_filename,
                                  temporal_weekend_matrix_filename,
                                  path_matrix_filename,
                                  path_weekeday_matrix_filename,
                                  path_weekend_matrix_filename,
                                  distance_matrix_filename,
                                  distance_weekday_matrix_filename,
                                  distance_weekend_matrix_filename,
                                  duration_matrix_filename,
                                  duration_weekday_matrix_filename,
                                  duration_weekend_matrix_filename,
                                  userid_column,
                                  category_column,
                                  locationid_column,
                                  latitude_column,
                                  longitude_column,
                                  datetime_column,
                                  differemt_venues,
                                  directed,
                                  personal_features_matrix,
                                  top_users,
                                  max_time_between_records,
                                  num_users,
                                  hour48=True,
                                  osm_category_column=None):

        if osm_category_column is not None:
            category_column = osm_category_column

        # shuffle
        users_checkin = users_checkin.sample(frac=1, random_state=1).reset_index(drop=True)

        users_checkin = users_checkin.dropna(subset=[userid_column, category_column, locationid_column, datetime_column])
        users_checkin = users_checkin.query(category_column + " != ''")
        ids = users_checkin[userid_column].unique().tolist()
        new_ids = []
        adj_matrices_column = []
        feat_matrices_column = []
        sequence_matrices_column = []
        categories_column = []
        count = 0
        # limitar usuarios
        print("us", len(ids))
        #users_checkin = users_checkin.query(userid_column + " in "+str(ids[:num_users]))
        selected_ids = users_checkin.groupby(userid_column).apply(lambda e: self.filter_user(e, userid_column, e[userid_column].iloc[0], datetime_column, category_column))
        selected_ids = selected_ids.query("tipo != 'nan'")
        selected_ids = selected_ids[userid_column].tolist()
        users_checkin = users_checkin.query(userid_column + " in " + str(selected_ids))
        print("Quantidade de usuários: ", len(selected_ids))
        start = time.time()
        users_checkin = users_checkin.groupby(userid_column).apply(lambda e: self.generate_user_matrices(e, e[userid_column].iloc[0],
                                                                                                         datetime_column,
                                                                                                         locationid_column,
                                                                                                         category_column,
                                                                                                         latitude_column,
                                                                                                         longitude_column,
                                                                                                         osm_category_column,
                                                                                                         differemt_venues, personal_features_matrix, hour48, directed, max_time_between_records))

        end = time.time()
        print("Duração: ", (end - start)/60)

        users_checkin = users_checkin.reset_index()
        print("checkis: \n", users_checkin)

        print("Filtro: ", count)

        print("tamanhos: ", len(new_ids), len(adj_matrices_column), len(categories_column))
        adjacency_matrix_df = users_checkin[['userid', 'adjacency', 'category']]
        adjacency_weekday_matrix_df = users_checkin[['userid', 'adjacency_weekday', 'category']]
        adjacency_weekend_matrix_df = users_checkin[['userid', 'adjacency_weekend', 'category']]
        temporal_matrix_df = users_checkin[['userid', 'temporal', 'category']]
        temporal_weekday_matrix_df = users_checkin[['userid', 'temporal_weekday', 'category']]
        temporal_weekend_matrix_df = users_checkin[['userid', 'temporal_weekend', 'category']]
        path_matrix_df = users_checkin[['userid', 'path', 'category']]
        path_weekeday_matrix_df = users_checkin[['userid', 'path_weekday', 'category']]
        path_weekend_matrix_df = users_checkin[['userid', 'path_weekend', 'category']]
        distance_matrix_df = users_checkin[['userid', 'distance', 'category']]
        distance_matrix_weekday_df = users_checkin[['userid', 'distance_weekday', 'category']]
        distance_matrix_weekend_df = users_checkin[['userid', 'distance_weekend', 'category']]
        duration_matrix_df = users_checkin[['userid', 'duration', 'category']]
        duration_matrix_weekeday_df = users_checkin[['userid', 'duration_weekday', 'category']]
        duration_matrix_weekend_df = users_checkin[['userid', 'duration_weekend', 'category']]

        adjacency_matrix_df.columns = ['user_id', 'matrices', 'category']
        adjacency_weekend_matrix_df.columns = ['user_id', 'matrices', 'category']
        adjacency_weekday_matrix_df.columns = ['user_id', 'matrices', 'category']
        temporal_matrix_df.columns = ['user_id', 'matrices', 'category']
        temporal_weekday_matrix_df.columns = ['user_id', 'matrices', 'category']
        temporal_weekend_matrix_df.columns = ['user_id', 'matrices', 'category']
        path_matrix_df.columns = ['user_id', 'matrices', 'category']
        path_weekeday_matrix_df.columns = ['user_id', 'matrices', 'category']
        path_weekend_matrix_df.columns = ['user_id', 'matrices', 'category']
        distance_matrix_df.columns = ['user_id', 'matrices', 'category']
        distance_matrix_weekday_df.columns = ['user_id', 'matrices', 'category']
        distance_matrix_weekend_df.columns = ['user_id', 'matrices', 'category']
        duration_matrix_df.columns = ['user_id', 'matrices', 'category']
        duration_matrix_weekeday_df.columns = ['user_id', 'matrices', 'category']
        duration_matrix_weekend_df.columns = ['user_id', 'matrices', 'category']
        # adjacency_matrix_df = pd.DataFrame(data={"user_id": new_ids,
        #                                         "matrices": adj_matrices_column,
        #                                         "category": categories_column })
        #
        # features_matrix_df = pd.DataFrame(data={"user_id": new_ids,
        #                                         "matrices": feat_matrices_column,
        #                                         "category": categories_column })
        #
        # sequence_matrix_df = pd.DataFrame(data={"user_id": new_ids,
        #                                         "matrices": sequence_matrices_column,
        #                                         "category": categories_column})

        files = [adjacency_matrix_df,
                adjacency_weekday_matrix_df,
                adjacency_weekend_matrix_df,
                temporal_matrix_df,
                temporal_weekday_matrix_df,
                temporal_weekend_matrix_df,
                path_matrix_df,
                path_weekeday_matrix_df,
                path_weekend_matrix_df,
                distance_matrix_df,
                distance_matrix_weekday_df,
                distance_matrix_weekend_df,
                duration_matrix_df,
                duration_matrix_weekeday_df,
                duration_matrix_weekend_df]

        files_names = [adjacency_matrix_filename,
                        adjacency_weekday_matrix_filename,
                        adjacency_weekend_matrix_filename,
                        temporal_matrix_filename,
                        temporal_weekday_matrix_filename,
                        temporal_weekend_matrix_filename,
                        path_matrix_filename,
                        path_weekeday_matrix_filename,
                        path_weekend_matrix_filename,
                       distance_matrix_filename,
                       distance_weekday_matrix_filename,
                       distance_weekend_matrix_filename,
                       duration_matrix_filename,
                       duration_weekday_matrix_filename,
                       duration_weekend_matrix_filename,
                       ]

        print(adjacency_matrix_df)
        self.matrix_generation_for_poi_categorization_loader.\
            adjacency_features_matrices_to_csv(files,
                                               files_names
                                               )

    def generate_gpr_matrices(self,
                                  users_checkin,
                                  adjacency_matrix_filename,
                                  features_matrix_filename,
                                  userid_column,
                                    latitude_column,
                                    longitude_column,
                                  category_column,
                                  locationid_column,
                                  datetime_column,
                                  directed,
                                  osm_category_column=None):

        print("pontos", osm_category_column)
        # constantes
        tempo_limite = 6

        if osm_category_column is not None:
            category_column = osm_category_column
        users_checkin = users_checkin
        users_checkin[datetime_column] = pd.to_datetime(users_checkin[datetime_column],
                                                        infer_datetime_format=True)

        users_checkin = users_checkin.dropna(
            subset=[userid_column, category_column, locationid_column, datetime_column])
        # shuffle
        users_checkin = users_checkin.sample(frac=1).reset_index(drop=True)
        ids = users_checkin[userid_column].unique().tolist()
        new_ids = []
        adj_matrices_column = []
        feat_matrices_column = []
        user_poi_vector_column = []
        categories_column = []
        count = 0
        it = 0
        for id_ in ids:
            #print(it)
            it += 1
            query = userid_column + "==" + "'" + str(id_) + "'"
            user_checkin = users_checkin.query(query)

            user_checkin = user_checkin.sort_values(by=[datetime_column])

            n_pois = len(user_checkin[locationid_column].unique().tolist())
            poi_poi_graph = [[0 for i in range(n_pois)] for j in range(n_pois)]
            features_matrix = [[0 for i in range(n_pois)] for j in range(n_pois)]
            # user-poi graph (vetor que contabiliza a quantidade de visitas em cada POI)
            user_poi_vector = [0] * n_pois
            categories_list = [-1 for i in range(n_pois)]

            datetimes = user_checkin[datetime_column].tolist()
            placeids = user_checkin[locationid_column].tolist()
            latitudes = user_checkin[latitude_column].tolist()
            longitudes = user_checkin[longitude_column].tolist()
            placeids_unique = user_checkin[locationid_column].unique().tolist()
            placeids_unique_to_int = {placeids_unique[i]: i for i in range(len(placeids_unique))}
            # converter os ids dos locais para inteiro
            placeids_int = [placeids_unique_to_int[placeids[i]] for i in range(len(placeids))]
            categories = user_checkin[category_column].tolist()
            # converter os nomes das categorias para inteiro
            # if osm_category_column is not None:
            #     # pre-processar raw gps
            #     categories = self.categories_list_preproccessing(categories, categories_to_int_osm)
            #
            # else:
            #     categories = self.categories_preproccessing(categories, categories_to_int_osm)

            if len(categories) < 2:
                continue

            # inicializar

            categories_list[0] = categories[0]
            user_poi_vector[placeids_int[0]] = 1

            for j in range(1, len(datetimes)):
                anterior = j - 1
                atual = j
                poi_anterior = placeids[anterior]
                poi_atual = placeids[atual]

                if (datetimes[atual] - datetimes[anterior]).total_seconds()/360 > tempo_limite:
                    continue

                if categories[atual] == "":
                    continue
                if directed:
                    poi_poi_graph[placeids_int[anterior]][placeids_int[atual]] += 1
                else:
                    poi_poi_graph[placeids_int[anterior]][placeids_int[atual]] += 1
                    poi_poi_graph[placeids_int[atual]][placeids_int[anterior]] += 1

                categories_list[placeids_int[atual]] = categories[atual]

                user_poi_vector[placeids_int[atual]] += 1

                # feature matrix (calcular distancia)
                if features_matrix[placeids_int[anterior]][placeids_int[atual]] != 0:
                    continue
                if features_matrix[placeids_int[anterior]][placeids_int[atual]] == 0 \
                        and placeids_int[anterior] != placeids_int[atual]:

                    di = points_distance([latitudes[anterior],
                                        longitudes[anterior]],
                                        [latitudes[atual],
                                        longitudes[atual]])

                    features_matrix[placeids_int[anterior]][placeids_int[atual]] = di
                    features_matrix[placeids_int[atual]][placeids_int[anterior]] = di

                # if osm_category_column is not None:
                #     # pre-processar raw gps
                #     poi_poi_graph, features_matrix, categories_list = self.remove_pois_that_dont_have_categories(
                #         categories_list, poi_poi_graph, features_matrix)
                #     if poi_poi_graph != []:
                #         count += 1

            poi_poi_graph, features_matrix, categories_list = self.remove_gpr_pois_that_dont_have_categories(
                categories_list, poi_poi_graph, features_matrix)
            if poi_poi_graph != []:
                count += 1

            if len(poi_poi_graph) <= 2 or len(features_matrix) <= 2 or len(categories_list) <= 2:
                continue

            new_ids.append(i)
            adj_matrices_column.append(str(poi_poi_graph))
            feat_matrices_column.append(str(features_matrix))
            user_poi_vector_column.append(str(user_poi_vector))
            categories_column.append(categories_list)

            # print("um usuario:\n", adjacency_matrix)

        print("Filtro: ", count)
        print("tamanhos: ", len(new_ids), len(adj_matrices_column), len(categories_column))
        adjacency_matrix_df = pd.DataFrame(data={"user_id": new_ids,
                                                 "matrices": adj_matrices_column,
                                                 "category": categories_column})

        features_matrix_df = pd.DataFrame(data={"user_id": new_ids,
                                                "matrices": feat_matrices_column,
                                                "category": categories_column})

        user_poi_matrix_df = pd.DataFrame(data={"user_id": new_ids,
                                                "vector": user_poi_vector_column})

        print(adjacency_matrix_df)
        adjacency_matrix_filename = adjacency_matrix_filename.replace("matrizes", "gpr")
        features_matrix_filename = features_matrix_filename.replace("matrizes", "gpr")

        print("nomee: ", features_matrix_filename, adjacency_matrix_filename)
        self.matrix_generation_for_poi_categorization_loader. \
            save_df_to_csv(adjacency_matrix_df, adjacency_matrix_filename)

        self.matrix_generation_for_poi_categorization_loader. \
            save_df_to_csv(features_matrix_df, features_matrix_filename)

        self.matrix_generation_for_poi_categorization_loader.\
            save_df_to_csv(user_poi_matrix_df, adjacency_matrix_filename.replace("adjacency_matrix", "user_poi_vector"))

    def _distance_between_pois(self, users_checkin, locationid_column,
                               latitude_column, longitude_column, n_pois):

        features_matrix = [[0 for i in range(n_pois)] for j in range(n_pois)]
        pois = users_checkin.groupby(locationid_column).apply(lambda e: e[[latitude_column, longitude_column]].iloc[0]).reset_index()
        print("pois: ", pois)

    def categories_preproccessing(self, categories, categories_to_int_osm):

        c = []
        for i in range(len(categories)):
            cate = categories_to_int_osm[categories[i].split(":")[0]]
            c.append(cate)

        return c

    def remove_gps_pois_that_dont_have_categories(self, categories, adjacency_matrix, features_matrix):

        indexes_filtered_pois = []
        adjacency_matrix = np.array(adjacency_matrix)
        features_matrix = np.array(features_matrix)
        for i in range(len(categories)):
            if categories[i] >= 0:
                indexes_filtered_pois.append(i)

        indexes_filtered_pois = np.array(indexes_filtered_pois)
        if len(indexes_filtered_pois) <= 1:
            return [], [], []

        categories = np.array(categories)
        categories = categories[indexes_filtered_pois]
        adjacency_matrix = adjacency_matrix[indexes_filtered_pois[:, None], indexes_filtered_pois]
        features_matrix = features_matrix[indexes_filtered_pois, :]

        if len(adjacency_matrix) <= 1:
            adjacency_matrix = []
            features_matrix = []

        return adjacency_matrix.tolist(), features_matrix.tolist(), categories.tolist()


    def remove_raw_gps_pois_that_dont_have_categories(self, categories, adjacency_matrix, features_matrix):

        indexes_filtered_pois = []
        adjacency_matrix = np.array(adjacency_matrix)
        features_matrix = np.array(features_matrix)
        for i in range(len(categories)):
            if len(categories[i]) >= 0 and categories[i][0] != "":
                indexes_filtered_pois.append(i)

        indexes_filtered_pois = np.array(indexes_filtered_pois)
        if len(indexes_filtered_pois) <= 1:
            return [], [], []

        categories = np.array(categories)
        categories = categories[indexes_filtered_pois]
        adjacency_matrix = adjacency_matrix[indexes_filtered_pois[:, None], indexes_filtered_pois]
        features_matrix = features_matrix[indexes_filtered_pois, :]

        if len(adjacency_matrix) <= 1:
            adjacency_matrix = []
            features_matrix = []

        return adjacency_matrix.tolist(), features_matrix.tolist(), categories.tolist()

    def remove_gpr_pois_that_dont_have_categories(self,
                                                  categories,
                                                  adjacency_matrix,
                                                  features_matrix):

        indexes_filtered_pois = []
        adjacency_matrix = np.array(adjacency_matrix)
        features_matrix = np.array(features_matrix)
        for i in range(len(categories)):
            if categories[i] >= 0:
                indexes_filtered_pois.append(i)

        indexes_filtered_pois = np.array(indexes_filtered_pois)
        if len(indexes_filtered_pois) <= 1:
            return [], [], []

        categories = np.array(categories)
        categories = categories[indexes_filtered_pois]
        adjacency_matrix = adjacency_matrix[indexes_filtered_pois[:, None], indexes_filtered_pois]
        features_matrix = features_matrix[indexes_filtered_pois, :]

        if len(adjacency_matrix) <= 1:
            adjacency_matrix = []
            features_matrix = []

        return adjacency_matrix.tolist(), features_matrix.tolist(), categories.tolist()

    def categories_list_preproccessing(self, categories, categories_to_int_osm):

        user_categories = []
        for i in range(len(categories)):
            c = []
            categories_names = categories[i].replace("'", "").replace(" ", "").replace("[", "").replace("]", "").split(",")
            print("element", categories_names)
            for category in categories_names:
                print("categorias: ", category)
                if category == "" or category == ' ':
                    c.append("")
                    continue
                cate = categories_to_int_osm[category]
                c.append(cate)
            user_categories.append(c)
        return user_categories

    def _summarize_categories_distance_matrix(self, categories_distances_matrix):
        sigma = 10
        categories_distances_list = []
        for row in range(len(categories_distances_matrix)):

            category_distances_list = []
            for column in range(len(categories_distances_matrix[row])):

                values = categories_distances_matrix[row][column]

                if len(values) == 0:
                    categories_distances_matrix[row][column] = 0
                    category_distances_list.append(0)
                else:

                    d_cc = st.median(values)
                    categories_distances_matrix[row][column] = d_cc

        return categories_distances_matrix

    def _duration_importance(self, duration):

        duration = duration * duration
        duration = -(duration / (self.duration_sigma * self.duration_sigma))
        duration = math.exp(duration)

        return duration

    def _distance_importance(self, distance):

        distance = distance * distance
        distance = -(distance / (self.distance_sigma * self.distance_sigma))
        distance = math.exp(distance)

        return distance
