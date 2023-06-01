import pandas as pd
import numpy as np
import json

import spektral.layers
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler, power_transform

import spektral as sk
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import sklearn.metrics as skm
from tensorflow.keras import utils as np_utils

from loader.file_loader import FileLoader
from loader.poi_categorization_loader import PoiCategorizationLoader
from extractor.file_extractor import FileExtractor
from model.neural_network.poi_gnn.BR.gnn_br_transfer_learning import GNNBR
from model.neural_network.poi_gnn.US.gnn import GNNUS
from model.neural_network.poi_gnn.path.gnn import GNNPath
from model.neural_network.poi_gnn.US.gnn_base_model_for_transfer_learning import GNNUS_BaseModel
from utils.nn_preprocessing import one_hot_decoding_predicted, top_k_rows, top_k_rows_category, top_k_rows_centrality, split_graph, top_k_rows_category_user_tracking, top_k_rows_order


class PoiCategorizationDomain:


    def __init__(self, dataset_name):
        self.file_loader = FileLoader()
        self.file_extractor = FileExtractor()
        self.poi_categorization_loader = PoiCategorizationLoader()
        self.dataset_name = dataset_name

    def _selecting_categories(self, categories: pd.Series):

        unique_category = categories.unique()
        unique_category = pd.Series(unique_category)
        unique_category = unique_category.str.split(":")
        unique_category = unique_category.tolist()
        first_category = []

        for c in unique_category:
            if type(c) == list:
                first_category.append(c[0][2:])

        unique_first_category = pd.Series(first_category)
        return unique_first_category

    def _first_level_categories(self, matrices: pd.Series, categories: pd.Series, categories_to_int_osm: dict):

        if self.dataset_name == "weeplaces":
            categories = categories.tolist()
            first_categories = []
            flatten = []
            unique_first_level_categories = list(categories_to_int_osm)
            print("unicos: ", unique_first_level_categories, type(unique_first_level_categories), unique_first_level_categories[1])
            for i in range(len(categories)):
                user_categories = categories[i]
                user_first_level_categories = []
                user_categories = user_categories.split(", ")
                inicial = len(user_categories)


                user_matrix = matrices.iloc[i]
                user_matrix = json.loads(user_matrix)
                user_matrix = np.array(user_matrix)

                if inicial != user_matrix.shape[0]:
                    print("Diferentes matrix: ", inicial, user_matrix.shape[0])
                for j in range(len(user_categories)):
                    element = user_categories[j]
                    element = element.replace("[", "").replace("]", "").replace("'", "")
                    for k in range(0, len(unique_first_level_categories)):
                        first_level_category = unique_first_level_categories[k]
                        if element.find(first_level_category) != -1:
                            element = first_level_category
                            break
                        if len(element) == 1 and element == ' ':
                            element = ''
                    user_first_level_categories.append(element)
                final = len(user_first_level_categories)
                if inicial != final:
                    print("Diferentes: ", inicial, final)
                flatten = flatten + user_first_level_categories
                first_categories.append(user_first_level_categories)
            print("categorias antes: ", pd.Series(flatten).unique().tolist())
            return first_categories
        else:
            categories = categories.tolist()
            first_categories = []

            for i in range(len(categories)):
                categories[i] = categories[i].replace("[", "").replace("]", "").replace("'","").split(", ")
                first_categories.append(categories[i])
            return first_categories

    def read_matrix(self, adjacency_matrix_filename, temporal_matrix_filename, distance_matrix_filename=None, duration_matrix_filename=None):

        adjacency_df = self.file_extractor.read_csv(adjacency_matrix_filename).drop_duplicates(subset=['user_id'])
        temporal_matrix_df = self.file_extractor.read_csv(temporal_matrix_filename).drop_duplicates(subset=['user_id'])
        if distance_matrix_filename is not None and duration_matrix_filename is not None:
            distance_matrix_df = self.file_extractor.read_csv(distance_matrix_filename).drop_duplicates(
                subset=['user_id'])
            duration_matrix_df = self.file_extractor.read_csv(duration_matrix_filename).drop_duplicates(subset=['user_id'])
            if adjacency_df['user_id'].tolist() != temporal_matrix_df['user_id'].tolist():
                print("MATRIZES DIFERENTES")
                raise

            return adjacency_df, temporal_matrix_df, distance_matrix_df, duration_matrix_df
        else:
            if adjacency_df['user_id'].tolist() != temporal_matrix_df['user_id'].tolist():
                print("MATRIZES DIFERENTES")
                raise
            return adjacency_df, temporal_matrix_df

    def read_matrices(self, adjacency_matrix_dir, feature_matrix_dir):

        adjacency_df = self.file_extractor.read_multiples_csv(adjacency_matrix_dir)
        feature_df = self.file_extractor.read_multiples_csv(feature_matrix_dir)

        return adjacency_df, feature_df

    def read_users_metrics(self, filename):

        return self.file_extractor.read_csv(filename).drop_duplicates(subset=['user_id'])

    def _poi_gnn_resize_adjacency_and_category_matrices(self, user_matrix, user_matrix_week, user_matrix_weekend, user_category, max_size_matrices, dataset_name):

        more_matrices = 1
        remain = False
        k_original = max_size_matrices
        size = user_matrix.shape[0]
        if size < k_original:
            k = user_matrix.shape[0]
        else:
            k = int(np.floor(size/k_original) * k_original)
        # select the k rows that have the highest sum
        if dataset_name == "user_tracking":
            idx = top_k_rows_category_user_tracking(user_matrix, k, user_category)
        else:
            idx = top_k_rows(user_matrix, k)

        not_used_ids = []
        for i in range(len(idx)):

            if i not in idx:
                not_used_ids.append(i)

        if len(not_used_ids) > 0 or size < max_size_matrices:

            add_more = max_size_matrices - len(not_used_ids)

            count = 0
            i = 0
            for i in idx:

                if count < add_more:

                    not_used_ids.append(i)
                    count += 1

                else:

                    break

        idx = np.array(idx.tolist() + not_used_ids)

        user_matrix = user_matrix[idx[:, None], idx]
        user_matrix_week = user_matrix_week[idx[:, None], idx]
        user_matrix_weekend = user_matrix_weekend[idx[:, None], idx]
        user_category = user_category[idx]

        if k > k_original or len(not_used_ids) > 0:
            k_split = int(np.floor(size/k_original))
            if len(not_used_ids) > 0:
                k_split += 1
            user_matrix = split_graph(user_matrix, k_original, k_split)
            user_matrix_week = split_graph(user_matrix_week, k_original, k_split)
            user_matrix_weekend = split_graph(user_matrix_weekend, k_original, k_split)
            user_category = split_graph(user_category, k_original, k_split)
            idx = split_graph(idx, k_original, k_split)
            more_matrices = k_split
            return np.array(user_matrix), np.array(user_matrix_week), np.array(user_matrix_weekend), np.array(
                user_category), np.array(idx), more_matrices

        return np.array([user_matrix]), np.array([user_matrix_week]), np.array([user_matrix_weekend]), np.array([user_category]), np.array([idx]), more_matrices

    def _resize_adjacency_and_category_matrices(self, user_matrix, user_matrix_week, user_matrix_weekend, user_category, max_size_matrices, dataset_name):

        k = max_size_matrices
        if user_matrix.shape[0] < k:
            k = user_matrix.shape[0]
        # select the k rows that have the highest sum
        if dataset_name == "user_tracking":
            idx = top_k_rows_category(user_matrix, k, user_category)
        else:
            idx = top_k_rows(user_matrix, k)
        user_matrix = user_matrix[idx[:,None], idx]
        user_matrix_week = user_matrix_week[idx[:, None], idx]
        user_matrix_weekend = user_matrix_weekend[idx[:, None], idx]
        user_category = user_category[idx]

        return user_matrix, user_matrix_week, user_matrix_weekend, user_category, idx

    def _resize_adjacency_and_category_matrices_baselines(self, user_matrix, user_category, max_size_matrices):

        k = max_size_matrices
        if user_matrix.shape[0] < k:
            k = user_matrix.shape[0]
        # select the k rows that have the highest sum
        idx = top_k_rows_order(user_matrix, k)
        user_matrix = user_matrix[idx[:,None], idx]
        user_category = user_category[idx]

        return user_matrix, user_category, idx

    def _resize_features_matrix(self, feature_matrix, idx, max_size_matrices):

        feature_matrix = feature_matrix[idx]
        if feature_matrix.shape[0] < max_size_matrices:
            difference = max_size_matrices - feature_matrix.shape[0]
            feature_matrix = np.pad(feature_matrix, (difference, 0), mode='constant', constant_values=0)
            feature_matrix = feature_matrix[:, :24]

        return feature_matrix

    def add_non_zero_indices(self, matrix, non_zero_indices, value):

        for i in range(len(non_zero_indices)):

            matrix[non_zero_indices[0][i]][non_zero_indices[1][i]] = matrix[non_zero_indices[0][i]][non_zero_indices[1][i]] + value

        return matrix

    def _augmentate_training_data(self, adjacency_matrices, features_matrices, categories, augmentation_cateogories: dict, user_metrics=None):

        new_adjacency_matrices = None
        new_features_matrices = None
        new_categories = None
        new_user_metrics = []
        matrix_shape = adjacency_matrices[0].shape
        print("tamaaanho", matrix_shape)


        adjacency_matrices = adjacency_matrices.tolist()
        features_matrices = features_matrices.tolist()
        categories = categories.tolist()

        for i in range(len(categories)):
            adjacency_matrix = adjacency_matrices[i]
            features_matrix = features_matrices[i]
            category = categories[i]
            if user_metrics is not None:
                user_metric = user_metrics[i]

            replication_level = 1

            if 0 in category or 6 in category:


                replication_level = 3


            # for j in range(len(augmentation_cateogories.keys())):
            #
            #     key = list(augmentation_cateogories.keys())[j]
            #     if key in category:
            #         if augmentation_cateogories[key] > replication_level:
            #             replication_level = augmentation_cateogories[key]

            # augmentation
            adjacency_non_zero_indices = np.nonzero(np.array(adjacency_matrix))
            features_non_zero_indices = np.nonzero(np.array(features_matrix))
            if replication_level > 1:
                aux = []
                aux2 = []
                aux3 = []
                for j in range(replication_level):
                    aux.append(category)
                    random_int = np.random.randint(low=2, high=10)
                    # aux2.append(self.add_non_zero_indices(matrix=adjacency_matrix,
                    #                                       non_zero_indices=adjacency_non_zero_indices, value=random_int))
                    # aux3.append(self.add_non_zero_indices(matrix=features_matrix,
                    #                                       non_zero_indices=features_non_zero_indices, value=random_int))

                    aux2.append(np.random.permutation(np.array(adjacency_matrix)).tolist())
                    aux3.append(np.random.permutation(np.array(features_matrix)).tolist())


                # for i in range(replication_level):
                #     aux.append(category)
                #     aux2.append(adjacency_matrix)
                #     aux3.append(features_matrix)
                #print("adj append", aux2)
                aux.append(category)
                category = aux
                aux2.append(adjacency_matrix)
                #print("adj append2", aux2)
                adjacency_matrix = aux2
                aux3.append(features_matrix)
                features_matrix = aux3
                #print("adj final: ", adjacency_matrix)
                if user_metrics is not None:
                    user_metric = user_metric * replication_level

            adjacency_matrix = np.array(adjacency_matrix)
            features_matrix = np.array(features_matrix)
            category = np.array(category)
            if adjacency_matrix.ndim == 2:
                adjacency_matrix = np.array([adjacency_matrix])
                features_matrix = np.array([features_matrix])

            if category.ndim == 1:
                category = np.array([category])

            if new_adjacency_matrices is None:
                new_adjacency_matrices = adjacency_matrix
                new_features_matrices = features_matrix
                new_categories = category
            else:
                new_adjacency_matrices = np.concatenate([new_adjacency_matrices, adjacency_matrix])
                new_features_matrices = np.concatenate([new_features_matrices, features_matrix])
                new_categories = np.concatenate([new_categories, category])

            if user_metrics is not None:
                new_user_metrics+= user_metric

        return new_adjacency_matrices, new_features_matrices, \
               new_categories, np.array(new_user_metrics)

    def _filter_pmi_matrix(self, location_time, location_location, locationid_to_int, visited_location_ids):

        idx = np.array([locationid_to_int[visited_location_ids[i]] for i in range(len(visited_location_ids))])

        location_time = location_time[idx]
        location_location = location_location[idx[:,None], idx].toarray()
        location_location = sk.layers.GCNConv.preprocess(location_location)

        return location_time, location_location

    def poi_gnn_adjacency_preprocessing(self,
                                inputs,
                                max_size_matrices,
                                week,
                                weekend,
                                num_categories,
                                dataset_name,
                                model_name="poi_gnn"):

        matrices_list = []
        temporal_matrices_list = []
        distance_matrices_list = []
        duration_matrices_list = []
        # week
        matrices_week_list = []
        temporal_matrices_week_list = []
        # weekend
        matrices_weekend_list = []
        temporal_matrices_weekend_list = []
        # location time
        location_time_list = []
        location_location_list = []

        users_categories = []
        flatten_users_categories = []
        maior = -10
        remove_users_ids = []

        matrix_df = inputs['all_week']['adjacency']
        ids = matrix_df['user_id'].unique().tolist()
        matrix_df = matrix_df['matrices'].tolist()
        category_df = inputs['all_week']['adjacency']['category'].tolist()
        temporal_df = inputs['all_week']['temporal']['matrices'].tolist()
        distance_df = inputs['all_week']['distance']['matrices'].tolist()
        duration_df = inputs['all_week']['duration']['matrices'].tolist()
        visited_location_ids = inputs['all_week']['adjacency']['visited_location_ids'].tolist()
        location_location_df = inputs['all_week']['location_location']
        location_time_df = inputs['all_week']['location_time'].to_numpy()
        locationid_to_int = inputs['all_week']['int_to_locationid']
        locationid_to_int_ids = locationid_to_int['locationid'].tolist()
        locationid_to_int_ints = locationid_to_int['int'].tolist()
        locationid_to_int = {locationid_to_int_ids[i]: locationid_to_int_ints[i] for i in range(len(locationid_to_int_ints))}
        # week
        matrix_week_df = inputs['week']['adjacency']['matrices'].tolist()
        temporal_week_df = inputs['week']['temporal']['matrices'].tolist()
        # weekend
        matrix_weekend_df = inputs['weekend']['adjacency']['matrices'].tolist()
        temporal_weekend_df = inputs['weekend']['temporal']['matrices'].tolist()

        selected_visited_locations = []


        if len(ids) != len(matrix_df):
            print("ERRO TAMANHO DA MATRIZ")
            exit()

        max_events = 0
        max_user = -1
        selected_users = []
        remove = 0
        for i in range(len(ids)):

            number_of_matrices = 1
            user_id = ids[i]

            user_matrices = matrix_df[i]
            user_category = category_df[i]
            user_matrices = json.loads(user_matrices)
            user_matrices = np.array(user_matrices)
            user_category = json.loads(user_category)
            user_category = np.array(user_category)
            # week
            user_matrices_week = matrix_week_df[i]
            user_matrices_week = json.loads(user_matrices_week)
            user_matrices_week = np.array(user_matrices_week)
            # weekend
            user_matrices_weekend = matrix_weekend_df[i]
            user_matrices_weekend = json.loads(user_matrices_weekend)
            user_matrices_weekend = np.array(user_matrices_weekend)
            # user visited
            user_visited = visited_location_ids[i]
            user_visited = json.loads(user_visited)
            user_visited = np.array(user_visited)
            # if user_matrices.shape[0] < max_size_matrices:
            #     remove_users_ids.append(user_id)
            #     remove += 1
            #     continue
            size = user_matrices.shape[0]
            if size > maior:
                maior = size

            # matrices get new size, equal for everyone
            user_matrices, user_matrices_week, user_matrices_weekend, user_category, idxs, number_of_matrices = self._poi_gnn_resize_adjacency_and_category_matrices(user_matrices, user_matrices_week, user_matrices_weekend, user_category, max_size_matrices, dataset_name)

            """feature"""
            user_temporal_matrices = temporal_df[i]
            user_temporal_matrices = json.loads(user_temporal_matrices)
            user_temporal_matrices = np.array(user_temporal_matrices)
            # week
            user_temporal_matrices_week = temporal_week_df[i]
            user_temporal_matrices_week = json.loads(user_temporal_matrices_week)
            user_temporal_matrices_week = np.array(user_temporal_matrices_week)
            # weekend
            user_temporal_matrices_weekend = temporal_weekend_df[i]
            user_temporal_matrices_weekend = json.loads(user_temporal_matrices_weekend)
            user_temporal_matrices_weekend = np.array(user_temporal_matrices_weekend)
            """distance"""
            user_distance_matrix = distance_df[i]
            user_distance_matrix = json.loads(user_distance_matrix)
            user_distance_matrix = np.array(user_distance_matrix)
            """duration"""
            user_duration_matrix = duration_df[i]
            user_duration_matrix = json.loads(user_duration_matrix)
            user_duration_matrix = np.array(user_duration_matrix)
            for i in range(number_of_matrices):
                idx = idxs[i]
                matrices_list.append(sk.layers.ARMAConv.preprocess(user_matrices[i]))
                matrices_week_list.append(sk.layers.ARMAConv.preprocess(user_matrices_week[i]))
                matrices_weekend_list.append(sk.layers.ARMAConv.preprocess(user_matrices_weekend[i]))

                user_temporal_matrix = user_temporal_matrices[idx]
                temporal_matrices_list.append(self._min_max_normalize(user_temporal_matrix))
                user_temporal_matrix_week = user_temporal_matrices_week[idx]
                temporal_matrices_week_list.append(self._min_max_normalize(user_temporal_matrix_week))
                user_temporal_matrix_weekend = user_temporal_matrices_weekend[idx]
                temporal_matrices_weekend_list.append(self._min_max_normalize(user_temporal_matrix_weekend))
                distance_matrices_list.append(user_distance_matrix[idx[:, None], idx])
                duration_matrices_list.append(user_duration_matrix[idx[:, None], idx])
                users_categories.append(user_category[i])
                # location time
                user_location_time, user_location_location = self._filter_pmi_matrix(location_time_df, location_location_df, locationid_to_int, user_visited[idx])
                user_location_time = self._min_max_normalize(user_location_time)
                location_time_list.append(user_location_time)
                user_location_location = spektral.layers.ARMAConv.preprocess(user_location_location)
                location_location_list.append(user_location_location)
                for j in user_visited[idx]:
                    selected_visited_locations.append(j)
                    selected_users.append(user_id)
            """"""
        df_selected_users_visited_locations = pd.DataFrame({'id': selected_users, 'poi_id': selected_visited_locations})
        print("lvisitas", len(df_selected_users_visited_locations))
        print("usuários", len(ids), " removidos: ", remove)
        self.features_num_columns = temporal_matrices_list[-1].shape[1]
        matrices_list = np.array(matrices_list)
        location_time_list = np.array(location_time_list)
        location_location_list = np.array(location_location_list)
        temporal_matrices_list = np.array(temporal_matrices_list)
        users_categories = np.array(users_categories)

        distance_matrices_list = np.array(distance_matrices_list)
        duration_matrices_list = np.array(duration_matrices_list)

        # week
        matrices_week_list = np.array(matrices_week_list)
        temporal_matrices_week_list = np.array(temporal_matrices_week_list)

        # weekend
        matrices_weekend_list = np.array(matrices_weekend_list)
        temporal_matrices_weekend_list = np.array(temporal_matrices_weekend_list)
        temporal_matrices_week_list = np.array(temporal_matrices_week_list)
        print("antes", matrices_list.shape, temporal_matrices_list.shape)

        print("Maior usuário: ", max_user, " ", max_events)


        print(users_categories.shape, matrices_list.shape, temporal_matrices_list.shape, distance_matrices_list.shape, duration_matrices_list.shape,
                        matrices_week_list.shape, temporal_matrices_week_list.shape, matrices_weekend_list.shape, temporal_matrices_weekend_list.shape,
                        location_time_list.shape, location_location_list.shape)
        return users_categories, matrices_list, temporal_matrices_list, distance_matrices_list, duration_matrices_list, \
               matrices_week_list, temporal_matrices_week_list, matrices_weekend_list, temporal_matrices_weekend_list, \
               location_time_list, location_location_list, selected_users, df_selected_users_visited_locations

    def adjacency_preprocessing(self,
                                inputs,
                                max_size_matrices,
                                max_size_sequence,
                                week,
                                weekend,
                                num_categories,
                                dataset_name,
                                model_name="poi_gnn"):

        matrices_list = []
        temporal_matrices_list = []
        distance_matrices_list = []
        duration_matrices_list = []
        # week
        matrices_week_list = []
        temporal_matrices_week_list = []
        # weekend
        matrices_weekend_list = []
        temporal_matrices_weekend_list = []
        # location time
        location_time_list = []
        location_location_list = []

        users_categories = []
        flatten_users_categories = []
        maior = -10
        remove_users_ids = []

        matrix_df = inputs['all_week']['adjacency']
        ids = matrix_df['user_id'].unique().tolist()
        matrix_df = matrix_df['matrices'].tolist()
        category_df = inputs['all_week']['adjacency']['category'].tolist()
        temporal_df = inputs['all_week']['temporal']['matrices'].tolist()

        if model_name == "poi_gnn":
            distance_df = inputs['all_week']['distance']['matrices'].tolist()
            duration_df = inputs['all_week']['duration']['matrices'].tolist()
            visited_location_ids = inputs['all_week']['adjacency']['visited_location_ids'].tolist()
            location_location_df = inputs['all_week']['location_location']
            location_time_df = inputs['all_week']['location_time'].to_numpy()
            locationid_to_int = inputs['all_week']['int_to_locationid']
            locationid_to_int_ids = locationid_to_int['locationid'].tolist()
            locationid_to_int_ints = locationid_to_int['int'].tolist()
            locationid_to_int = {locationid_to_int_ids[i]: locationid_to_int_ints[i] for i in range(len(locationid_to_int_ints))}
            # week
            matrix_week_df = inputs['week']['adjacency']['matrices'].tolist()
            temporal_week_df = inputs['week']['temporal']['matrices'].tolist()
            # weekend
            matrix_weekend_df = inputs['weekend']['adjacency']['matrices'].tolist()
            temporal_weekend_df = inputs['weekend']['temporal']['matrices'].tolist()


        if len(ids) != len(matrix_df):
            print("ERRO TAMANHO DA MATRIZ")
            exit()

        max_events = 0
        max_user = -1
        selected_users = []
        for i in range(len(ids)):
            # if i >= 1000:
            #     continue
            #print("indice: ", i)
            user_id = ids[i]
            # if user_id not in users_metrics_ids:
            #     continue
            #     print("diferentes", user_id)
            #     remove_users_ids.append(user_id)

            user_matrix = matrix_df[i]
            user_category = category_df[i]
            user_matrix = json.loads(user_matrix)
            user_matrix = np.array(user_matrix)
            user_category = json.loads(user_category)
            user_category = np.array(user_category)
            if model_name == "poi_gnn":
                # week
                user_matrix_week = matrix_week_df[i]
                user_matrix_week = json.loads(user_matrix_week)
                user_matrix_week = np.array(user_matrix_week)
                # weekend
                user_matrix_weekend = matrix_weekend_df[i]
                user_matrix_weekend = json.loads(user_matrix_weekend)
                user_matrix_weekend = np.array(user_matrix_weekend)
                # user visited
                user_visited = visited_location_ids[i]
                user_visited = json.loads(user_visited)
                user_visited = np.array(user_visited)
            if user_matrix.shape[0] < max_size_matrices:
                remove_users_ids.append(user_id)
                continue
            size = user_matrix.shape[0]
            if size > maior:
                maior = size

            # matrices get new size, equal for everyone
            if model_name == "poi_gnn":
                if week and weekend:
                    user_matrix, user_matrix_week, user_matrix_weekend, user_category, idx = self._resize_adjacency_and_category_matrices(user_matrix, user_matrix_week, user_matrix_weekend, user_category, max_size_matrices, dataset_name)
                    #print("user matrix week: ", user_matrix_week)
                    user_total = np.sum(user_matrix)
                    if user_total > max_events:
                        max_events = user_total
                        max_user = i
                else:
                    user_matrix, user_category, idx = self._resize_adjacency_and_category_matrices_baselines(
                        user_matrix, user_category, max_size_matrices)
                user_visited = user_visited[idx]
                if model_name == "gcn" or model_name == "gae":
                    user_matrix = sk.layers.GCNConv.preprocess(user_matrix)
                elif model_name == "arma" or model_name == "arma_enhanced" or model_name == "poi_gnn":
                    user_matrix = sk.layers.ARMAConv.preprocess(user_matrix)
                    user_matrix_week = sk.layers.ARMAConv.preprocess(user_matrix_week)
                    user_matrix_weekend = sk.layers.ARMAConv.preprocess(user_matrix_weekend)
                elif model_name == "diff":
                    user_matrix = sk.layers.DiffusionConv.preprocess(user_matrix)
            else:
                user_matrix, user_category, idx = self._resize_adjacency_and_category_matrices_baselines(
                    user_matrix, user_category, max_size_matrices)
                if model_name == "gcn" or model_name == "gae":
                    user_matrix = sk.layers.GCNConv.preprocess(user_matrix)
                elif model_name == "arma" or model_name == "arma_enhanced" or model_name == "poi_gnn":
                    user_matrix = sk.layers.ARMAConv.preprocess(user_matrix)
                elif model_name == "diff":
                    user_matrix = sk.layers.DiffusionConv.preprocess(user_matrix)

            # if len(pd.Series(user_category).unique().tolist()) < num_categories - 1:
            #     print("parou")
            #     continue

            """feature"""
            user_temporal_matrix = temporal_df[i]
            user_temporal_matrix = json.loads(user_temporal_matrix)
            user_temporal_matrix = np.array(user_temporal_matrix)
            user_temporal_matrix = user_temporal_matrix[idx]
            user_temporal_matrix = self._min_max_normalize(user_temporal_matrix)
            if model_name == "poi_gnn":
                # week
                user_temporal_matrix_week = temporal_week_df[i]
                user_temporal_matrix_week = json.loads(user_temporal_matrix_week)
                user_temporal_matrix_week = np.array(user_temporal_matrix_week)
                user_temporal_matrix_week = user_temporal_matrix_week[idx]
                # weekend
                user_temporal_matrix_weekend = temporal_weekend_df[i]
                user_temporal_matrix_weekend = json.loads(user_temporal_matrix_weekend)
                user_temporal_matrix_weekend = np.array(user_temporal_matrix_weekend)
                user_temporal_matrix_weekend = user_temporal_matrix_weekend[idx]
                # location time
                user_location_time, user_location_location = self._filter_pmi_matrix(location_time_df, location_location_df, locationid_to_int, user_visited)
                user_location_time = self._min_max_normalize(user_location_time)
                location_time_list.append(user_location_time)
                user_location_location = spektral.layers.ARMAConv.preprocess(user_location_location)
                location_location_list.append(user_location_location)

            if model_name == "poi_gnn":
                """distance"""
                user_distance_matrix = distance_df[i]
                user_distance_matrix = json.loads(user_distance_matrix)
                user_distance_matrix = np.array(user_distance_matrix)
                user_distance_matrix = user_distance_matrix[idx[:,None], idx]

                """duration"""
                user_duration_matrix = duration_df[i]
                user_duration_matrix = json.loads(user_duration_matrix)
                user_duration_matrix = np.array(user_duration_matrix)
                user_duration_matrix = user_duration_matrix[idx[:,None], idx]

            """"""
            matrices_list.append(user_matrix)
            users_categories.append(user_category)
            flatten_users_categories = flatten_users_categories + user_category.tolist()
            temporal_matrices_list.append(user_temporal_matrix)
            if model_name == "poi_gnn":
                distance_matrices_list.append(user_distance_matrix)
                duration_matrices_list.append(user_duration_matrix)
                # week
                matrices_week_list.append(user_matrix_week)
                temporal_matrices_week_list.append(user_temporal_matrix_week)
                # weekend
                matrices_weekend_list.append(user_matrix_weekend)
                temporal_matrices_weekend_list.append(user_temporal_matrix_weekend)

            selected_users.append(user_id)

        self.features_num_columns = temporal_matrices_list[-1].shape[1]
        matrices_list = np.array(matrices_list)
        location_time_list = np.array(location_time_list)
        location_location_list = np.array(location_location_list)
        temporal_matrices_list = np.array(temporal_matrices_list)
        users_categories = np.array(users_categories)
        if model_name == "poi_gnn":
            distance_matrices_list = np.array(distance_matrices_list)
            duration_matrices_list = np.array(duration_matrices_list)

            # week
            matrices_week_list = np.array(matrices_week_list)
            temporal_matrices_week_list = np.array(temporal_matrices_week_list)

            # weekend
            matrices_weekend_list = np.array(matrices_weekend_list)
            temporal_matrices_weekend_list = np.array(temporal_matrices_weekend_list)
            temporal_matrices_week_list = np.array(temporal_matrices_week_list)
        print("antes", matrices_list.shape, temporal_matrices_list.shape)

        print("Maior usuário: ", max_user, " ", max_events)

        if model_name == "poi_gnn":
            if week and weekend:
                return (users_categories, matrices_list, temporal_matrices_list, distance_matrices_list, duration_matrices_list,
                        matrices_week_list, temporal_matrices_week_list, matrices_weekend_list, temporal_matrices_weekend_list,
                        location_time_list, location_location_list, selected_users)
            else:
                return (matrices_list, users_categories, temporal_matrices_list,distance_matrices_list,
                        duration_matrices_list, remove_users_ids, selected_users)
        else:
            return (matrices_list, users_categories, temporal_matrices_list)

    def generate_nodes_ids(self, rows, cols):


        ids = []
        for i in range(rows):
            row = [i for i in range(cols)]
            ids.append(row)

        return np.array(ids)

    def k_fold_split_train_test(self,
                                k,
                                inputs,
                                n_splits,
                                week_type,
                                model_name='poi_gnn'):

        adjacency_list = inputs[week_type]['adjacency']
        temporal_list = inputs[week_type]['temporal']
        user_categories = inputs[week_type]['categories']
        if model_name == "poi_gnn" and week_type == 'all_week':
            distance_list = inputs[week_type]['distance']
            duration_list = inputs[week_type]['duration']
            location_time = inputs[week_type]['location_time']
            location_location_list = inputs[week_type]['location_location']
        else:
            distance_list = []
            duration_list = []
            location_time = []
            location_location_list = []
        skip = False
        if n_splits == 1:
            skip = True
            n_splits = 2
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=0)

        folds = []
        classes_weights = []
        for train_indexes, test_indexes in kf.split(adjacency_list):

            fold, class_weight = self._split_train_test(k,
                                                        model_name,
                                                        adjacency_list,
                                                        user_categories,
                                                        temporal_list,
                                                        location_time,
                                                        location_location_list,
                                                        distance_list,
                                                        duration_list,
                                                        train_indexes,
                                                        test_indexes)
            folds.append(fold)
            classes_weights.append(class_weight)
            if skip:
                break

        return folds, classes_weights

    def preprocess_adjacency_matrix_train(self, k, model_name, adjacency_list_train, user_categories_train, temporal_list_train, distance_list_train, duration_list_train):

        new_adjacency_list_train = []
        new_user_categories_train = []
        new_temporal_list_train = []
        new_distance_list_train = []
        new_duration_list_train = []
        print("tamanho: ", len(adjacency_list_train), adjacency_list_train.shape)
        for i in range(len(adjacency_list_train)):

            adjacency_train = adjacency_list_train[i]
            category = user_categories_train[i]
            temporal = temporal_list_train[i]

            idx = top_k_rows(adjacency_train, k)

            adjacency_train = adjacency_train[idx[:,None], idx]
            category = category[idx]
            temporal = temporal[idx]
            new_user_categories_train.append(category)
            new_temporal_list_train.append(temporal)

            if len(distance_list_train) > 0:

                distance = distance_list_train[i]
                duration = duration_list_train[i]
                distance = distance[idx[:,None], idx]
                duration = duration[idx[:,None], idx]
                new_distance_list_train.append(distance)
                new_duration_list_train.append(duration)

            if model_name == "gcn" or model_name == "gae":
                adjacency_train = sk.layers.GCNConv.preprocess(adjacency_train)
            elif model_name == "arma" or model_name == "arma_enhanced" or model_name == "poi_gnn":
                adjacency_train = sk.layers.ARMAConv.preprocess(adjacency_train)
                user_matrix_week = sk.layers.ARMAConv.preprocess(adjacency_train)
                user_matrix_weekend = sk.layers.ARMAConv.preprocess(adjacency_train)
            elif model_name == "diff":
                user_matrix = sk.layers.DiffusionConv.preprocess(adjacency_train)

            new_adjacency_list_train.append(adjacency_train)

        return np.array(new_adjacency_list_train), np.array(new_user_categories_train), np.array(new_temporal_list_train), np.array(new_distance_list_train), np.array(new_duration_list_train)

    def preprocess_adjacency_matrix_test(self, k, model_name, adjacency_list_test, user_categories_test, temporal_list_test, distance_list_test, duration_list_test):

        new_adjacency_list_test = []
        new_user_categories_test = []
        new_temporal_list_test = []
        new_distance_list_test = []
        new_duration_list_test = []
        for i in range(len(adjacency_list_test)):

            adjacency_test = adjacency_list_test[i]
            category = user_categories_test[i]
            temporal = temporal_list_test[i]

            idx = top_k_rows(adjacency_test, k)

            adjacency_test = adjacency_test[idx[:,None], idx]
            category = category[idx]
            temporal = temporal[idx]
            new_user_categories_test.append(category)
            new_temporal_list_test.append(temporal)

            if len(distance_list_test) > 0:

                distance = distance_list_test[i]
                duration = duration_list_test[i]
                distance = distance[idx[:,None], idx]
                duration = duration[idx[:,None], idx]
                new_distance_list_test.append(distance)
                new_duration_list_test.append(duration)

            if model_name == "gcn" or model_name == "gae":
                adjacency_test = sk.layers.GCNConv.preprocess(adjacency_test)
            elif model_name == "arma" or model_name == "arma_enhanced" or model_name == "poi_gnn":
                adjacency_train = sk.layers.ARMAConv.preprocess(adjacency_test)
                user_matrix_week = sk.layers.ARMAConv.preprocess(adjacency_test)
                user_matrix_weekend = sk.layers.ARMAConv.preprocess(adjacency_test)
            elif model_name == "diff":
                user_matrix = sk.layers.DiffusionConv.preprocess(adjacency_test)

            new_adjacency_list_test.append(adjacency_test)

        return np.array(new_adjacency_list_test), np.array(new_user_categories_test), np.array(new_temporal_list_test), np.array(new_distance_list_test), np.array(new_duration_list_test)


    def _split_train_test(self,
                          k,
                          model_name,
                          adjacency_list,
                          user_categories,
                          temporal_list,
                          location_time_list,
                          location_location_list,
                          distance_list,
                          duration_list,
                          train_indexes,
                          test_indexes):

        size = adjacency_list.shape[0]
        # 'average', 'cv', 'median', 'radius', 'label'
        adjacency_list_train = adjacency_list[train_indexes]
        user_categories_train = user_categories[train_indexes]

        temporal_list_train = temporal_list[train_indexes]

        if len(distance_list) > 0:
            distance_list_train = distance_list[train_indexes]
            duration_list_train = duration_list[train_indexes]
            location_time_list_train = location_time_list[train_indexes]
            location_location_list_train = location_location_list[train_indexes]
        else:
            distance_list_train = []
            duration_list_train = []
            location_time_list_train = []
            location_location_list_train = []

        # adjacency_list_train, user_categories_train, temporal_list_train, distance_list_train, duration_list_train = self.\
        #     preprocess_adjacency_matrix_train(k,
        #                                       model_name,
        #                                       adjacency_list_train,
        #                                       user_categories_train,
        #                                       temporal_list_train,
        #                                       distance_list_train, duration_list_train)


        adjacency_list_test = adjacency_list[test_indexes]
        user_categories_test = user_categories[test_indexes]
        temporal_list_test = temporal_list[test_indexes]

        if len(distance_list) > 0:
            distance_list_test = distance_list[test_indexes]
            duration_list_test = duration_list[test_indexes]
            location_time_list_test = location_time_list[test_indexes]
            location_location_list_test = location_location_list[test_indexes]
        else:
            distance_list_test = []
            duration_list_test = []
            location_time_list_test = []
            location_location_list_test = []

        # adjacency_list_test, user_categories_test, temporal_list_test, distance_list_test, duration_list_test = self.\
        #     preprocess_adjacency_matrix_test(k,
        #                                      model_name,
        #                                      adjacency_list_test,
        #                                      user_categories_test,
        #                                      temporal_list_test,
        #                                      distance_list_test,
        #                                      duration_list_test)

        flatten_train_category = []
        for categories_list in user_categories_train:
            flatten_train_category += categories_list.tolist()
        flatten_train_category = pd.Series(flatten_train_category, name='category')
        flatten_train_category = flatten_train_category.astype('object')
        train_categories_freq = {e:0 for e in flatten_train_category.unique().tolist()}
        for i in range(flatten_train_category.shape[0]):
            train_categories_freq[flatten_train_category.iloc[i]]+=1
        n = sum(train_categories_freq.values())

        total_support = 0
        for e in train_categories_freq:
            total_support+=train_categories_freq[e]

        total_support_inverse = 0
        for e in train_categories_freq:
            total_support_inverse += total_support - train_categories_freq[e]


        for e in train_categories_freq:
            train_categories_freq[e] = (total_support - train_categories_freq[e])/total_support_inverse


        class_weight = list(train_categories_freq.values())
        user_categories_train = np.array([[e for e in row] for row in user_categories_train])
        user_categories_test = np.array([[e for e in row] for row in user_categories_test])

        if len(distance_list) > 0:
            return (adjacency_list_train, user_categories_train, temporal_list_train, distance_list_train, duration_list_train,
                    location_time_list_train, location_location_list_train, adjacency_list_test, user_categories_test, temporal_list_test, distance_list_test,
                    duration_list_test, location_time_list_test, location_location_list_test), class_weight
        else:
            print("retornoo")
            return (adjacency_list_train, user_categories_train, temporal_list_train,
                    adjacency_list_test, user_categories_test, temporal_list_test), class_weight


    def _preprocess_features(self, features):
        # print("antes: ", features[0])
        # rowsum = np.array(features.sum(1))
        # r_inv = np.power(rowsum, 1).flatten()
        # r_inv[np.isinf(r_inv)] = 0.
        # r_mat_inv = sp.diags(r_inv)
        # features = r_mat_inv.dot(features)
        # print("depois: ", features[0])
        features = normalize(features, axis=1)
        return features

    def k_fold_with_replication_train_and_evaluate_model(self,
                                                         inputs_folds,
                                                         n_replications,
                                                         max_size_matrices,
                                                         max_size_sequence,
                                                         base_report,
                                                         epochs,
                                                         class_weight,
                                                         base,
                                                         country,
                                                         version,
                                                         output_dir):

        folds_histories = []
        folds_reports = []
        models = []
        accuracies = []
        seed = 0
        for i in range(len(inputs_folds['all_week']['folds'])):

            fold = inputs_folds['all_week']['folds'][i]
            fold_week = inputs_folds['week']['folds'][i]
            fold_weekend = inputs_folds['weekend']['folds'][i]
            class_weight = inputs_folds['all_week']['class_weight'][i]
            class_weight_week = inputs_folds['week']['class_weight'][i]
            class_weight_weekend = inputs_folds['weekend']['class_weight'][i]
            histories = []
            reports = []
            for j in range(n_replications):

                history, report, model, accuracy = self.train_and_evaluate_model(i,
                                                                                 fold,
                                                                                 fold_week,
                                                                                 fold_weekend,
                                                                                class_weight,
                                                                                 class_weight_week,
                                                                                 class_weight_weekend,
                                                                                max_size_matrices,
                                                                                max_size_sequence,
                                                                                epochs,
                                                                                seed,
                                                                                country,
                                                                                 base,
                                                                                output_dir,
                                                                                version)

                seed+=1

                base_report = self._add_location_report(base_report, report)
                histories.append(history)
                reports.append(report)
                models.append(model)
                accuracies.append(accuracy)
            folds_histories.append(histories)
            folds_reports.append(reports)
        best_model = self._find_best_model(models, accuracies)

        return folds_histories, base_report, best_model

    def train_and_evaluate_model(self,
                                 fold_number,
                                 fold,
                                 fold_week,
                                 fold_weekend,
                                 class_weight,
                                 class_weight_week,
                                 class_weight_weekend,
                                 max_size_matrices,
                                 max_size_sequence,
                                 epochs,
                                 seed,
                                 country,
                                 base,
                                 output_dir,
                                 version="normal",
                                 model=None):


        #print("entradas: ", adjacency_train.shape, features_train.shape, y_train.shape)
        #print("enstrada test: ", adjacency_test.shape, features_test.shape, y_test.shape)
        adjacency_train, y_train, temporal_train, distance_train, duration_train,  location_time_train, location_location_train,\
        adjacency_test, y_test, temporal_test, distance_test, duration_test, location_time_test, location_location_test = fold
        adjacency_week_train, y_train_week, temporal_train_week,  \
        adjacency_test_week, y_test_week, temporal_test_week = fold_week
        adjacency_train_weekend, y_train_weekend, temporal_train_weekend, \
        adjacency_test_weekend, y_test_weekend, temporal_test_weekend = fold_weekend

        max_total = 0
        max_user = -1

        for i in range(len(adjacency_test)):
            user_total = np.sum(adjacency_test[i])
            if user_total > max_total:
                max_total = user_total
                max_user = i

        num_classes = max(y_train.flatten()) + 1
        max_size = max_size_matrices
        lr = 0.001
        print("Quantidade de classes: ", num_classes)
        print("Tamanho maximo", max_size_matrices)
        print("Tamanho das matrizes de treino: ", adjacency_train.shape, temporal_train.shape,
              adjacency_week_train.shape, temporal_train_week.shape, distance_train.shape, duration_train.shape, location_time_train.shape, location_location_train.shape)

        print("Tamanho das matrizes de teste: ", adjacency_test.shape, temporal_test.shape,
              adjacency_test_week.shape, temporal_test_week.shape, distance_test.shape, duration_test.shape, location_time_test.shape, location_location_train.shape)
        if country == 'BR' or country == "Brazil":
            if version == "normal":
                print("Tipo de rede neural: NORMAL")
                model = GNNBR(num_classes, max_size, max_size_sequence,
                            self.features_num_columns).build(seed=seed)
                lr = 0.005
                epochs = 35
            elif version == "PATH":
                print("PATH")
                model = GNNPath(num_classes, max_size, max_size_sequence,
                            self.features_num_columns).build(seed=seed)
        elif country == 'US':
            if base:
                print("dentro")
                model = GNNUS_BaseModel(num_classes, max_size, max_size_sequence,
                              self.features_num_columns).build(seed=seed)
            else:
                model = GNNUS_BaseModel(num_classes, max_size, max_size_sequence,
                        self.features_num_columns).build(seed=seed)
        if country == 'US':
            batch = max_size * 2
        elif country == 'BR' or country == 'Brazil':
            batch = 1

        print("Tamanho do batch: ", batch)

        user_index = max_user
        self.heatmap_matrices(str(fold_number), [adjacency_test[user_index], adjacency_test_week[user_index], adjacency_test_weekend[user_index],
                               temporal_test[user_index], temporal_test_week[user_index], temporal_test_weekend[user_index], location_time_test[user_index], location_location_test[user_index]],
                              ["Adjacency", "Adjacency (weekday)", "Adjacency (weekend)", "Temporal", "Temporal (weekday)", "Temporal (weekend)", "Location_time", "Location_location"],
                              output_dir)

        input_train = [adjacency_train, adjacency_week_train, adjacency_train_weekend,
                       temporal_train, temporal_train_week, temporal_train_weekend, distance_train,
                       duration_train, location_time_train, location_location_train]
        input_test = [adjacency_test, adjacency_test_week, adjacency_test_weekend, temporal_test, temporal_test_week, temporal_test_weekend,
                      distance_test, duration_test, location_time_test, location_location_test]


        # verifying whether categories arrays are equal
        compare1 = y_train == y_train_week
        compare2 = y_train_week == y_train_weekend
        compare3 = y_test == y_test_week
        compare4 = y_test_week == y_test_weekend
        if not(compare1.all() and compare2.all() and compare3.all() and compare4.all()):
            print("Listas difernetes de categorias")
            exit()

        model.compile(optimizer=Adam(learning_rate=lr), loss=['categorical_crossentropy'],
                      weighted_metrics=[tf.keras.metrics.CategoricalAccuracy(name="acc")
                                        ])
        y_train = np_utils.to_categorical(y_train, num_classes=num_classes)
        y_test = np_utils.to_categorical(y_test, num_classes=num_classes)
        hi = model.fit(x=input_train,
                       y=y_train, validation_data=(input_test, y_test),
                       epochs=epochs, batch_size=batch,
                       shuffle=False,  # Shuffling data means shuffling the whole graph
                       callbacks=[
                           EarlyStopping(patience=100, restore_best_weights=True)
                       ]
                       )

        h = hi.history
        #print("summary: ", model.summary())
        y_predict_location = model.predict(input_test,
                                           batch_size=batch)

        scores = model.evaluate(input_test,
                                y_test, batch_size=batch)
        print("scores: ", scores)

        # To transform one_hot_encoding to list of integers, representing the locations
        y_predict_location = one_hot_decoding_predicted(y_predict_location)
        y_test = one_hot_decoding_predicted(y_test)
        report = skm.classification_report(y_test, y_predict_location, output_dict=True)
        print(report)

        return h, report, model, report['accuracy']

    def heatmap_matrices(self, fold_number, matrices, names, output_dir):

        for matrix, name in zip(matrices, names):

            self.poi_categorization_loader.heatmap(output_dir, matrix, name.replace(" ", "_")+"_"+fold_number, name, (10,10), True)

    def _add_location_report(self, location_report, report):
        for l_key in report:
            if l_key == 'accuracy':
                location_report[l_key].append(report[l_key])
                continue
            for v_key in report[l_key]:
                location_report[l_key][v_key].append(report[l_key][v_key])

        return location_report

    def _find_best_model(self, models, accuracies):

        index = np.argmax(accuracies)
        return models[index]

    def preprocess_report(self, report, int_to_categories):

        new_report = {}

        for key in report:
            if key != 'accuracy' and key != 'macro avg' and key != 'weighted avg':
                new_report[int_to_categories[key]] = report[key]
            else:
                new_report[key] = report[key]

        return new_report

    def _min_max_normalize(self, matrix):

        matrix_1 = matrix.transpose()
        scaler = MinMaxScaler()
        scaler.fit(matrix_1)
        matrix_1 = scaler.transform(matrix_1).transpose()


        # min_value = matrix.min()
        # max_value = matrix.max()
        #
        # matrix = (matrix-min_value)/(max_value-min_value)
        # matrix += matrix_1

        return matrix_1

        