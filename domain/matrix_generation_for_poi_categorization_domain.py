import math
import pandas as pd
import numpy as np
import time
import statistics as st
from scipy.sparse import dok_matrix
from numpy.linalg import norm
from numpy.linalg import inv as inverse
import scipy.sparse as sparse
from sklearn.decomposition import NMF
from configuration.weekday  import Weekday

from foundation.util.geospatial_utils import points_distance
from loader.matrix_generation_for_poi_categorization_loarder import MatrixGenerationForPoiCategorizationLoader
from extractor.file_extractor import FileExtractor
from foundation.util.statistics_utils import pmi
import skmob

from skmob.measures.individual import k_radius_of_gyration


class MatrixGenerationForPoiCategorizationDomain:


    def __init__(self, dataset_name):
        self.matrix_generation_for_poi_categorization_loader = MatrixGenerationForPoiCategorizationLoader()
        self.file_extractor = FileExtractor()
        self.dataset_name = dataset_name
        self.distance_sigma = 10
        self.duration_sigma = 10
        self.max_events = 200
        self.LL = np.array([])
        self.LT = np.array([])

    def filter_user(self, user_checkin,
                    dataset_name,
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
        if dataset_name == "gowalla":
            if len(user_checkin[category_column].unique().tolist()) < 7:
                return pd.DataFrame({'tipo': ['nan'], userid_column: [userid]})
        elif dataset_name == "user_tracking":
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

        user_checkin = user_checkin.sort_values(by=[datetime_column])
        #user_checkin = user_checkin.head(self.max_events + 200)
        latitude_list = user_checkin[latitude_column].tolist()
        longitude_list = user_checkin[longitude_column].tolist()

        # matrices initialization
        visited_location_ids = user_checkin[locationid_column].unique().tolist()
        n_pois = len(visited_location_ids)
        adjacency_matrix = [[0 for i in range(n_pois)] for j in range(n_pois)]
        adjacency_weekday_matrix = [[0 for i in range(n_pois)] for j in range(n_pois)]
        adjacency_weekend_matrix = [[0 for i in range(n_pois)] for j in range(n_pois)]
        temporal_weekday_matrix = [[0 for i in range(24)] for j in range(n_pois)]
        temporal_weekend_matrix = [[0 for i in range(24)] for j in range(n_pois)]
        distance_matrix = [[0 for i in range(n_pois)] for j in range(n_pois)]
        duration_matrix = [[[] for i in range(n_pois)] for j in range(n_pois)]
        if personal_features_matrix or not hour48:
            temporal_matrix = [[0 for i in range(24)] for j in range(n_pois)]
        else:
            temporal_matrix = [[0 for i in range(48)] for j in range(n_pois)]
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

        if len(user_checkin[category_column].unique().tolist()) < 5:
            print("Usuário com poucas categorias diferentes visitadas")
            return pd.DataFrame({'adjacency': ['vazio'], 'adjacency_weekday': ['vazio'],
                                 'adjacency_weekend': ['vazio'], 'temporal': ['vazio'],
                                 'distance': ['vazio'], 'duration': ['vazio'],
                                 'temporal_weekday': ['vazio'],
                                 'temporal_weekend': ['vazio'],
                                 'visited_location_ids': ['vazio'],
                                 'category': ['vazio']})

        # matrices initialization - setting the first visit
        if not personal_features_matrix:
            if hour48:
                if datetimes[0].weekday() < 5:
                    hour = datetimes[0].hour
                    temporal_matrix[placeids_int[0]][hour] += 1
                    temporal_weekday_matrix[placeids_int[0]][hour] += 1
                else:
                    hour = datetimes[0].hour + 24
                    temporal_matrix[placeids_int[0]][hour] += 1
                    temporal_weekend_matrix[placeids_int[0]][hour - 24] += 1
            else:
                hour = datetimes[0].hour
                temporal_matrix[placeids_int[0]][hour] += 1
        else:
            if datetimes[0].weekday() < 5:
                temporal_matrix[placeids_int[0]][math.floor(datetimes[0].hour / 2)] += 1
            else:
                temporal_matrix[placeids_int[0]][math.floor(datetimes[0].hour / 2) + 12] += 1
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
            if distance_matrix[local_anterior][local_atual] == 0:
                distance = int(points_distance([lat_before, lng_before], [lat_current, lng_current]) / 1000)
                distance = self._distance_importance(distance)
            else:
                distance = distance_matrix[local_anterior][local_atual]

            datetime_before = datetimes[anterior]
            datetime_current = datetimes[atual]
            duration = int((datetime_current - datetime_before).total_seconds() / 3600)
            duration = self._duration_importance(duration)
            distance_matrix[local_anterior][local_atual] = distance
            distance_matrix[local_atual][local_anterior] = distance
            duration_matrix[local_anterior][local_atual].append(duration)

            if directed:
                adjacency_matrix[placeids_int[anterior]][placeids_int[atual]] += 1
                if datetimes[atual].weekday() < 5:
                    adjacency_weekday_matrix[placeids_int[anterior]][placeids_int[atual]] += 1
                else:
                    adjacency_weekend_matrix[placeids_int[anterior]][placeids_int[atual]] += 1
            else:
                adjacency_matrix[placeids_int[anterior]][placeids_int[atual]] += 1
                adjacency_matrix[placeids_int[atual]][placeids_int[anterior]] += 1

                if datetimes[atual].weekday() < 5:
                    adjacency_weekday_matrix[placeids_int[anterior]][placeids_int[atual]] += 1
                    adjacency_weekday_matrix[placeids_int[atual]][placeids_int[anterior]] += 1
                else:
                    adjacency_weekend_matrix[placeids_int[anterior]][placeids_int[atual]] += 1
                    adjacency_weekend_matrix[placeids_int[atual]][placeids_int[anterior]] += 1

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


        duration_matrix = self._summarize_categories_distance_matrix(duration_matrix)

        visited_location_ids = pd.Series(visited_location_ids).unique().tolist()

        if len(pd.Series(categories_list).unique().tolist()) < 5:
            print("Usuário com poucas categorias diferentes visitadas")
            return pd.DataFrame({'adjacency': ['vazio'], 'adjacency_weekday': ['vazio'],
                                 'adjacency_weekend': ['vazio'], 'temporal': ['vazio'],
                                 'distance': ['vazio'], 'duration': ['vazio'],
                                 'temporal_weekday': ['vazio'],
                                 'temporal_weekend': ['vazio'],
                                 'visited_location_ids': ['vazio'],
                                 'category': ['vazio']})

        return pd.DataFrame({'adjacency': [adjacency_matrix], 'adjacency_weekday': [adjacency_weekday_matrix],
                             'adjacency_weekend': [adjacency_weekend_matrix], 'temporal': [temporal_matrix],
                             'distance': [distance_matrix], 'duration': [duration_matrix],
                             'temporal_weekday': [temporal_weekday_matrix], 'temporal_weekend': [temporal_weekend_matrix],
                             'visited_location_ids': [visited_location_ids],
                             'category': [categories_list]})

    def generate_gpr_user_matrices(self, user_checkin,
                               userid,
                               datetime_column,
                               locationid_column,
                               category_column,
                               latitude_column,
                               longitude_column,
                               osm_category_column,
                               differemt_venues, max_time_between_records):
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

        user_checkin = user_checkin.sort_values(by=[datetime_column])
        user_checkin = user_checkin.head(self.max_events + 200)
        latitude_list = user_checkin[latitude_column].tolist()
        longitude_list = user_checkin[longitude_column].tolist()

        # matrices initialization
        n_pois = len(user_checkin[locationid_column].unique().tolist())
        adjacency_matrix = [[0 for i in range(n_pois)] for j in range(n_pois)]
        adjacency_weekday_matrix = [[0 for i in range(n_pois)] for j in range(n_pois)]
        adjacency_weekend_matrix = [[0 for i in range(n_pois)] for j in range(n_pois)]
        distance_matrix = [[0 for i in range(n_pois)] for j in range(n_pois)]
        poi_poi_graph = [[0 for i in range(n_pois)] for j in range(n_pois)]
        user_poi_vector_column = []
        user_poi_vector = [0] * n_pois
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

        if len(user_checkin[category_column].unique().tolist()) < 5:
            print("Usuário com poucas categorias diferentes visitadas")
            return pd.DataFrame({'adjacency': ['vazio'], 'adjacency_weekday': ['vazio'],
                                 'adjacency_weekend': ['vazio'], 'temporal': ['vazio'],
                                 'distance': ['vazio'],
                                 'category': ['vazio']})

        # matrices initialization - setting the first visit

        categories_list[0] = categories[0]
        user_poi_vector[placeids_int[0]] = 1

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

            user_poi_vector[placeids_int[atual]] += 1
            poi_poi_graph[placeids_int[anterior]][placeids_int[atual]] += 1
            lat_before = latitude_list[anterior]
            lng_before = longitude_list[anterior]
            lat_current = latitude_list[atual]
            lng_current = longitude_list[atual]
            if distance_matrix[local_anterior][local_atual] == 0:
                distance = int(points_distance([lat_before, lng_before], [lat_current, lng_current]) / 1000)
                distance = self._distance_importance(distance)
            else:
                distance = distance_matrix[local_anterior][local_atual]

            datetime_before = datetimes[anterior]
            datetime_current = datetimes[atual]
            duration = int((datetime_current - datetime_before).total_seconds() / 3600)
            duration = self._duration_importance(duration)
            distance_matrix[local_anterior][local_atual] = distance
            distance_matrix[local_atual][local_anterior] = distance

            adjacency_matrix[placeids_int[anterior]][placeids_int[atual]] += 1
            if datetimes[atual].weekday() < 5:
                adjacency_weekday_matrix[placeids_int[anterior]][placeids_int[atual]] += 1
            else:
                adjacency_weekend_matrix[placeids_int[anterior]][placeids_int[atual]] += 1

            categories_list[placeids_int[atual]] = categories[atual]

            if len(poi_poi_graph) <= 2 or len(distance_matrix) <= 2 or len(categories_list) <= 2:
                continue

        poi_poi_graph, distance_matrix, categories_list = self.remove_gpr_pois_that_dont_have_categories(
            categories_list, poi_poi_graph, distance_matrix)
        if poi_poi_graph != []:
            count += 1




        if len(pd.Series(categories_list).unique().tolist()) < 5:
            print("Usuário com poucas categorias diferentes visitadas")
            return pd.DataFrame({'adjacency': ['vazio'],
                                 'distance': ['vazio'],
                                 'category': ['vazio']})

        return pd.DataFrame({'adjacency': [adjacency_matrix],
                             'distance': [distance_matrix],
                             'category': [categories_list]})

    def reduce_user_data(self, user_checkin, datetime_column):

        user_checkin = user_checkin.sort_values(by=[datetime_column])
        user_checkin = user_checkin.head(self.max_events + 200)

        return user_checkin

    def _create_location_coocurrency_matrix(self, users_checkins, userid_column, datetime_column, locationid_column, locationid_to_int):
        try:

            users_checkins["time"] = [d.time() for d in users_checkins[datetime_column]]
            number_of_locations = len(users_checkins[locationid_column].unique())
            self.LL = sparse.lil_matrix(
                (number_of_locations, number_of_locations))  ##location co occurency represents memory for save memory
            self.LL_radius = [{} for i in range(number_of_locations)]
            df_LL_user_frequency = [[] for i in range(number_of_locations)]
            df_LL_frequency = [[] for i in range(number_of_locations)]
            df_LL_users = [[] for i in range(number_of_locations)]

            cont = 0
            init = time.time()
            for user_id in users_checkins[userid_column].unique():
                if cont % 100 == 0:
                    end = time.time()
                    print("usuário: ", cont)
                    print("duração: ", (end - init)/60)

                cont += 1
                users_checkins_sorted = users_checkins[users_checkins[userid_column] == user_id].sort_values(by=[datetime_column])
                locations_frequency = users_checkins_sorted.groupby(locationid_column).count().reset_index()[[locationid_column, 'userid']]
                locations_frequency.columns = np.array([locationid_column, 'count'])
                locations_frequency['count'] = locations_frequency['count'] / len(locations_frequency)
                locations_frequency['count'] = locations_frequency['count'].round(5)
                user_locations_frequency = locations_frequency[locationid_column].tolist()
                user_locations_frequency_count = locations_frequency['count'].tolist()
                for j in range(len(user_locations_frequency)):
                    current_location_frequency = locationid_to_int[user_locations_frequency[j]]
                    count = user_locations_frequency_count[j]
                    df_LL_user_frequency[current_location_frequency].append(count)
                # locations = users_checkins_sorted[locationid_column].tolist()
                # latitudes = users_checkins_sorted['latitude'].tolist()
                # longitudes = users_checkins_sorted['longitude'].tolist()
                # 
                # previous_location = locations[0]
                # for i in range(len(locations)):
                #     current_location = locationid_to_int[locations[i]]
                #     if i > 0 and i < len(locations) -1:
                #         future_location = locations[i+1]
                #         if len(self.LL_radius[current_location]) == 0:
                #             self.LL_radius[current_location] = {previous_location, locations[i], future_location}
                #         else:
                #             self.LL_radius[current_location].update({previous_location, future_location})
                #     for j in range(1, 6):
                #         if ((i - j) < 0):
                #             break
                #         self.LL[current_location, locationid_to_int[locations[i - j]]] += 1
                #     for j in range(1, 6):
                #         if (i + j) > len(locations) - 1:
                #             break
                #         self.LL[current_location, locationid_to_int[locations[j + i]]] += 1

            for i in range(len(df_LL_user_frequency)):
                if len(df_LL_user_frequency[i]) > 1:
                    df_LL_user_frequency[i] = [round(st.mean(df_LL_user_frequency[i]), 5), round(st.median(df_LL_user_frequency[i]), 5), round(st.stdev(df_LL_user_frequency[i]), 5)]
                else:
                    df_LL_user_frequency[i] = [df_LL_user_frequency[i][0], df_LL_user_frequency[i][0], 0]

            df_LL_user_frequency = pd.DataFrame(df_LL_user_frequency, columns=['average_users_frequency', 'median_users_frequency', 'stdev_users_frequency'])

            # locations_frequency = users_checkins.groupby(locationid_column).count().reset_index()[[locationid_column, 'userid']]
            # locations_frequency.columns = np.array([locationid_column, 'count'])
            # locations = locations_frequency[locationid_column].tolist()
            # counts = locations_frequency['count'].tolist()
            # total = sum(counts)
            # for i in range(len(locations)):
            #     location = locationid_to_int[locations[i]]
            #     count = counts[i]
            #     df_LL_frequency[location] = round(count/total, 5)
            #
            # print(df_LL_frequency)
            
            locations_users = users_checkins.groupby([locationid_column, userid_column]).count().reset_index()[[locationid_column, userid_column]].groupby(locationid_column).count().reset_index()
            locations_users.columns = np.array([locationid_column, 'count'])
            locations = locations_users[locationid_column].tolist()
            counts = locations_users['count'].tolist()
            for i in range(len(locations)):
                location = locationid_to_int[locations[i]]
                count = counts[i]
                df_LL_users[location] = count

            df_LL_users = pd.DataFrame(df_LL_users, columns=['distinct_users'])
            ll = pd.concat([df_LL_user_frequency, df_LL_users], axis=1)
            print(ll)
            self.LL = ll
            # df = None
            # init = time.time()
            # for i in range(len(self.LL_radius)):
            #
            #     location_trajectory = users_checkins[users_checkins[locationid_column].isin(list(self.LL_radius[i]))][[datetime_column, 'latitude', 'longitude']]
            #     location_trajectory['poi_in'] = np.array([i for i in range(len(location_trajectory))])
            #     if df is None:
            #         df = location_trajectory
            #     else:
            #         df = pd.concat([df, location_trajectory], ignore_index=True)
            #
            # tdf = skmob.TrajDataFrame(df, latitude='latitude', longitude='longitude', datetime=datetime_column,
            #                           user_id='poi_int')
            #
            # krg_df = k_radius_of_gyration(tdf, k=20)
            # end = time.time()
            # print(krg_df)
            # print("Duração raio de giro: ", (end - init)/60)
            # exit()

            # init = time.time()
            # sum_of_dl = self.LL.sum()
            # l_occurrency = self.LL.sum(axis=1)
            # c_occurrency = self.LL.sum(axis=0)
            # end = time.time()
            # print("Preencheu a matriz localização x localização", (end - init)/60)
            # init = time.time()
            # 
            # 
            # row, column = self.LL.nonzero()
            # for i, j in zip(row, column):
            # 
            #     try:
            #         p = (self.LL[i, j] * number_of_locations) / ( l_occurrency[i, 0] * c_occurrency[0, j])
            #         np.nan_to_num(p, copy=False, nan=0)
            #         p = np.maximum(p, 1)
            #         self.LL[i, j] = np.maximum(np.log2(p), 0)
            #     except:
            #         print(self.LL[i, j], number_of_locations)
            #         print(l_occurrency[i])
            #         print(c_occurrency)
            #         print(c_occurrency[0, j])
            # for i in range(number_of_locations):
            #     line = self.LL[i].toarray()
            #     p = (line * sum_of_dl) / (l_occurrency[i] * c_occurrency)
            #     p[p == np.inf] = 0
            #     np.nan_to_num(p, copy=False, nan=0)
            #     p = np.maximum(p, 1)
            #     self.LL[i] = np.maximum(np.log2(p), 0)
            end = time.time()
            print("calculou os totais", (end - init)/60)
            ##No sparse implementation
            """ Dl = np.zeros((number_of_locations, number_of_locations))
            for i in range(len(locations)):
                for j in range(1, 6):
                    if((i - j) < 0):
                        break
                    Dl[locations[i], locations[i - j]] += 1
                for j in range(1, 6):
                    if(i + j) > len(locations) - 1:
                        break
                    Dl[locations[i], locations[j + i]] += 1
            sum_of_dl = np.sum(Dl)
            l_occurrency = np.sum(Dl, axis=1).reshape(-1, 1)
            c_occurrency = np.sum(Dl, axis=0).reshape(1, -1)
            teste = np.maximum(np.log2(np.maximum(Dl * sum_of_dl, 1)/(l_occurrency * c_occurrency)), 0)

            print((self.LL[1].toarray() == teste[1]).all())
            print("FIM")         """
        except Exception as e:
            raise e

    def _create_LT_matrix(self, users_checinks, locationid_column, datetime_column, locationid_to_int):

        locations = users_checinks[locationid_column].tolist()
        datetimes = users_checinks[datetime_column].tolist()
        unique_locationsids = users_checinks[locationid_column].unique().tolist()
        total_locations = len(unique_locationsids)
        Dt = np.zeros((total_locations, 48))

        for i in range(len(locations)):
            current_location = locationid_to_int[locations[i]]
            if (datetimes[i].weekday() >= Weekday.SATURDAY.value):
                Dt[current_location][datetimes[i].hour + 24] += 1
            else:
                Dt[current_location][datetimes[i].hour] += 1

        sum_of_dt = np.sum(Dt)
        l_occurrency = np.sum(Dt, axis=1).reshape(-1, 1)
        c_occurrency = np.sum(Dt, axis=0).reshape(1, -1)

        with np.errstate(divide='ignore'):
            p = (Dt * sum_of_dt) / (l_occurrency * c_occurrency)
            p = np.maximum(p, 1)
            self.LT = np.maximum(np.log2(p), 0)

    def define_pmi_matrices(self, users_checkins, userid_column, datetime_column, locationid_column, max_time_between_records):

        print("matriz pmi")
        print(users_checkins)
        print(datetime_column)
        n_max_pois = 141900
        datetimes = users_checkins[datetime_column].tolist()
        location_ids = users_checkins[locationid_column].tolist()
        unique_location_ids = users_checkins[locationid_column].unique().tolist()
        ids = users_checkins[userid_column].unique().tolist()

        location_ids_to_matrix_index = {unique_location_ids[i]: i for i in range(len(unique_location_ids))}
        users_pois_ids = {i: [] for i in ids}
        #location_location_pmi_matrix = dok_matrix((n_max_pois, n_max_pois), dtype=np.float32)
        location_time_pmi_matrix = dok_matrix((n_max_pois, 48), dtype=np.float32)


        count = 0
        max_timedelta = pd.Timedelta(days=2020)
        if len(max_time_between_records) > 0:
            max_timedelta = pd.Timedelta(days=int(max_time_between_records))
        for j in range(1, len(datetimes)):
            anterior = j - 1
            atual = j
            local_anterior = location_ids[anterior]
            local_atual = location_ids[atual]
            # retirar eventos muito esparços
            if len(max_time_between_records) > 0:
                if (datetimes[atual] - datetimes[anterior]) > max_timedelta:
                    continue
            # retirar eventos consecutivos em um mesmo estabelecimento

            if local_anterior == local_atual:
                continue

            current_datetime = datetimes[atual]
            hour = current_datetime.hour
            if current_datetime.weekday() > 4:
                hour = hour + 24
            #location_location_pmi_matrix[location_ids_to_matrix_index[local_anterior], location_ids_to_matrix_index[local_atual]] += 1
            location_time_pmi_matrix[location_ids_to_matrix_index[local_atual], hour] += 1

        print("L T")
        print(location_time_pmi_matrix)
        exit()

        # total_location_i = [sum(location_location_pmi_matrix[i]) for i in range(len(location_location_pmi_matrix))]
        # total_location_j = [sum(location_location_pmi_matrix[:, i]) for i in range(len(location_location_pmi_matrix))]
        total_hour = [sum(location_time_pmi_matrix[:, i]) for i in range(len(location_time_pmi_matrix[0]))]

        # for i in range(len(location_location_pmi_matrix)):
        #
        #     for j in range(len(location_location_pmi_matrix[i])):
        #         location_location_pmi_matrix[i, j] = pmi(location_location_pmi_matrix[i, j], n_max_pois, total_location_i[i], total_location_j[j])

        for i in range(len(location_time_pmi_matrix)):

            for j in range(len(location_time_pmi_matrix[i])):
                #location_location_pmi_matrix[i, j] = pmi(location_time_pmi_matrix[i, j], n_max_pois, total_location_i[i], total_hour[j])
                pass

        #print(location_location_pmi_matrix)

        #df = pd.DataFrame({'location_location_pmi': [list(location_location_pmi_matrix)], 'location_time_pmi': [list(location_time_pmi_matrix)], 'poi_id': [list(location_ids_to_matrix_index.keys())], 'poi_index': [list(location_ids_to_matrix_index.values())]})

        #print(df)
        print("l")
        exit()
        #return df


    def generate_pattern_matrices(self,
                                  users_checkin,
                                  dataset_name,
                                  adjacency_matrix_filename,
                                  adjacency_weekday_matrix_filename,
                                  adjacency_weekend_matrix_filename,
                                  temporal_matrix_filename,
                                  temporal_weekday_matrix_filename,
                                  temporal_weekend_matrix_filename,
                                  distance_matrix_filename,
                                  duration_matrix_filename,
                                  location_location_pmi_matrix_filename,
                                  location_time_omi_matrix_filename,
                                  int_to_locationid_filename,
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
        print(users_checkin)
        users_checkin = users_checkin.dropna(subset=[userid_column, category_column, locationid_column, datetime_column])
        users_checkin = users_checkin.query(category_column + " != ''")
        ids = users_checkin.groupby('userid').count().sort_values('placeid', ascending=False).reset_index()['userid'].tolist()
        #ids = users_checkin.groupby('userid').count().reset_index().sample(frac=1, random_state=1)['userid'].tolist()
        #ids = users_checkin[userid_column].unique().tolist()
        new_ids = []
        adj_matrices_column = []
        feat_matrices_column = []
        sequence_matrices_column = []
        categories_column = []
        count = 0
        # limitar usuarios
        print("us", len(ids))
        num_users = 1000
        users_checkin = users_checkin.query(userid_column + " in "+str(ids[:num_users]))
        # selected_ids = users_checkin.groupby(userid_column).apply(lambda e: self.filter_user(e, dataset_name, userid_column, e[userid_column].iloc[0], datetime_column, category_column))
        # selected_ids = selected_ids.query("tipo != 'nan'")
        # selected_ids = selected_ids[userid_column].tolist()
        # users_checkin = users_checkin.query(userid_column + " in " + str(selected_ids))
        # print("Quantidade de usuários: ", len(selected_ids))
        start = time.time()
        users_checkin['userid'] = users_checkin[userid_column].to_numpy()
        users_checkin[locationid_column] = users_checkin[locationid_column].astype(int)
        original_columns = users_checkin.columns.tolist()
        users_checkin = users_checkin.groupby('userid').apply(lambda e: self.reduce_user_data(e, datetime_column)).rename(columns={'userid': 'index'}).reset_index()[original_columns]
        print("depois")
        print(users_checkin)
        unique_locationsids = users_checkin[locationid_column].unique().tolist()
        locationid_to_int = {unique_locationsids[i]: i for i in range(len(unique_locationsids))}
        keys = list(locationid_to_int.keys())
        values = list(locationid_to_int.values())
        int_to_location_id = {values[i]: keys[i] for i in range(len(keys))}
        categories = [users_checkin[users_checkin[locationid_column] == unique_locationsids[i]][category_column].iloc[0] for i in range(len(unique_locationsids))]
        print("categorias")
        self._create_LT_matrix(users_checkin, locationid_column, datetime_column, locationid_to_int)
        print("terminou LT")
        lt = pd.DataFrame(self.LT, columns=[str(i) for i in range(self.LT.shape[1])])
        lt['category'] = np.array(categories)
        self.matrix_generation_for_poi_categorization_loader.save_df_to_csv(lt, location_time_omi_matrix_filename)
        self.matrix_generation_for_poi_categorization_loader.save_df_to_csv(pd.DataFrame({'locationid': keys, 'int': values}), int_to_locationid_filename)
        lt = ""
        self.LT = ""
        self._create_location_coocurrency_matrix(users_checkin, userid_column, datetime_column, locationid_column, locationid_to_int)
        print("terminou LL")
        #self.matrix_generation_for_poi_categorization_loader.save_sparse_matrix_to_npz(sparse.csr_matrix(self.LL), location_location_pmi_matrix_filename)
        self.matrix_generation_for_poi_categorization_loader.save_df_to_csv(self.LL, location_location_pmi_matrix_filename.replace("npz", "csv"))
        self.LL = ""
        exit()
        users_checkin = users_checkin.groupby('userid').apply(lambda e: self.generate_user_matrices(e, e['userid'].iloc[0],
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
        users_checkin = users_checkin[users_checkin['adjacency'] != 'vazio']
        adjacency_matrix_df = users_checkin[['userid', 'adjacency', 'category', 'visited_location_ids']]
        adjacency_weekday_matrix_df = users_checkin[['userid', 'adjacency_weekday', 'category']]
        adjacency_weekend_matrix_df = users_checkin[['userid', 'adjacency_weekend', 'category']]
        temporal_matrix_df = users_checkin[['userid', 'temporal', 'category']]
        temporal_weekday_matrix_df = users_checkin[['userid', 'temporal_weekday', 'category']]
        temporal_weekend_matrix_df = users_checkin[['userid', 'temporal_weekend', 'category']]
        distance_matrix_df = users_checkin[['userid', 'distance', 'category']]
        duration_matrix_df = users_checkin[['userid', 'duration', 'category']]

        adjacency_matrix_df.columns = ['user_id', 'matrices', 'category', 'visited_location_ids']
        adjacency_weekend_matrix_df.columns = ['user_id', 'matrices', 'category']
        adjacency_weekday_matrix_df.columns = ['user_id', 'matrices', 'category']
        temporal_matrix_df.columns = ['user_id', 'matrices', 'category']
        temporal_weekday_matrix_df.columns = ['user_id', 'matrices', 'category']
        temporal_weekend_matrix_df.columns = ['user_id', 'matrices', 'category']
        distance_matrix_df.columns = ['user_id', 'matrices', 'category']
        duration_matrix_df.columns = ['user_id', 'matrices', 'category']
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
                distance_matrix_df,
                duration_matrix_df]

        files_names = [adjacency_matrix_filename,
                        adjacency_weekday_matrix_filename,
                        adjacency_weekend_matrix_filename,
                        temporal_matrix_filename,
                        temporal_weekday_matrix_filename,
                        temporal_weekend_matrix_filename,
                       distance_matrix_filename,
                       duration_matrix_filename
                       ]

        print(adjacency_matrix_df)
        self.matrix_generation_for_poi_categorization_loader.\
            adjacency_features_matrices_to_csv(files,
                                               files_names
                                               )

    def generate_gpr_matrices_v2(self,
                                  users_checkin,
                                  dataset_name,
                                  adjacency_matrix_filename,
                                  adjacency_weekday_matrix_filename,
                                  adjacency_weekend_matrix_filename,
                                  distance_matrix_filename,
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
        print(users_checkin)
        users_checkin = users_checkin.dropna(subset=[userid_column, category_column, locationid_column, datetime_column])
        users_checkin = users_checkin.query(category_column + " != ''")
        ids = users_checkin.groupby('userid').count().sort_values('placeid', ascending=False).reset_index()['userid'].tolist()
        #ids = users_checkin.groupby('userid').count().reset_index().sample(frac=1, random_state=1)['userid'].tolist()
        #ids = users_checkin[userid_column].unique().tolist()
        new_ids = []
        adj_matrices_column = []
        feat_matrices_column = []
        sequence_matrices_column = []
        categories_column = []
        count = 0
        # limitar usuarios
        print("us", len(ids))

        users_checkin = users_checkin.query(userid_column + " in "+str(ids[:10000]))
        # selected_ids = users_checkin.groupby(userid_column).apply(lambda e: self.filter_user(e, dataset_name, userid_column, e[userid_column].iloc[0], datetime_column, category_column))
        # selected_ids = selected_ids.query("tipo != 'nan'")
        # selected_ids = selected_ids[userid_column].tolist()
        # users_checkin = users_checkin.query(userid_column + " in " + str(selected_ids))
        # print("Quantidade de usuários: ", len(selected_ids))
        start = time.time()
        users_checkin['userid'] = users_checkin[userid_column].to_numpy()
        users_checkin = users_checkin.groupby('userid').apply(lambda e: self.generate_gpr_user_matrices(e, e['userid'].iloc[0],
                                                                                                         datetime_column,
                                                                                                         locationid_column,
                                                                                                         category_column,
                                                                                                         latitude_column,
                                                                                                         longitude_column,
                                                                                                         osm_category_column,
                                                                                                         differemt_venues, max_time_between_records))

        end = time.time()
        print("Duração: ", (end - start)/60)

        users_checkin = users_checkin.reset_index()
        print("checkis: \n", users_checkin)

        print("Filtro: ", count)

        print("tamanhos: ", len(new_ids), len(adj_matrices_column), len(categories_column))
        users_checkin = users_checkin[users_checkin['adjacency'] != 'vazio']
        adjacency_matrix_df = users_checkin[['userid', 'adjacency', 'category']]
        adjacency_weekday_matrix_df = users_checkin[['userid', 'adjacency_weekday', 'category']]
        adjacency_weekend_matrix_df = users_checkin[['userid', 'adjacency_weekend', 'category']]
        distance_matrix_df = users_checkin[['userid', 'distance', 'category']]

        adjacency_matrix_df.columns = ['user_id', 'matrices', 'category']
        adjacency_weekend_matrix_df.columns = ['user_id', 'matrices', 'category']
        adjacency_weekday_matrix_df.columns = ['user_id', 'matrices', 'category']
        distance_matrix_df.columns = ['user_id', 'matrices', 'category']
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
                distance_matrix_df]

        files_names = [adjacency_matrix_filename,
                        adjacency_weekday_matrix_filename,
                        adjacency_weekend_matrix_filename,
                       distance_matrix_filename
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
