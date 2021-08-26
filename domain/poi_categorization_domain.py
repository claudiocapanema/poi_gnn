import pandas as pd
import numpy as np
import json
from sklearn.model_selection import KFold

import spektral as sk
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
import sklearn.metrics as skm
from tensorflow.keras import utils as np_utils

from loader.file_loader import FileLoader
from extractor.file_extractor import FileExtractor
from model.neural_network.poi_gnn.BR.gnn import GNN
from model.neural_network.poi_gnn.US.gnn import GNNUS
from model.neural_network.poi_gnn.path.gnn import GNNPath
from utils.nn_preprocessing import one_hot_decoding_predicted, top_k_rows


class PoiCategorizationDomain:


    def __init__(self, dataset_name):
        self.file_loader = FileLoader()
        self.file_extractor = FileExtractor()
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

    def read_matrix(self, adjacency_matrix_filename, feature_matrix_filename, distance_matrix_filename=None, duration_matrix_filename=None):

        adjacency_df = self.file_extractor.read_csv(adjacency_matrix_filename).drop_duplicates(subset=['user_id'])
        feature_df = self.file_extractor.read_csv(feature_matrix_filename).drop_duplicates(subset=['user_id'])
        if adjacency_df['user_id'].tolist() != feature_df['user_id'].tolist():
            print("MATRIZES DIFERENTES")

        distance_matrix_df = None
        if distance_matrix_filename is not None:
            distance_matrix_df = self.file_extractor.read_csv(distance_matrix_filename).drop_duplicates(
                subset=['user_id'])

        duration_matrix_df = None
        if duration_matrix_filename is not None:
            print("duracao", duration_matrix_filename)
            duration_matrix_df = self.file_extractor.read_csv(duration_matrix_filename).drop_duplicates(
                subset=['user_id'])

        return adjacency_df, feature_df, distance_matrix_df, duration_matrix_df

    def read_matrices(self, adjacency_matrix_dir, feature_matrix_dir):

        adjacency_df = self.file_extractor.read_multiples_csv(adjacency_matrix_dir)
        feature_df = self.file_extractor.read_multiples_csv(feature_matrix_dir)

        return adjacency_df, feature_df

    def read_users_metrics(self, filename):

        return self.file_extractor.read_csv(filename).drop_duplicates(subset=['user_id'])

    def _resize_adjacency_and_category_matrices(self, user_matrix, user_matrix_week, user_matrix_weekend, user_category, max_size_matrices):

        k = max_size_matrices
        if user_matrix.shape[0] < k:
            k = user_matrix.shape[0]
        # select the k rows that have the highest sum
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
        idx = top_k_rows(user_matrix, k)
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


    def adjacency_preprocessing(self,
                                inputs,
                                max_size_matrices,
                                max_size_sequence,
                                week,
                                weekend,
                                model_name="poi_gnn"):

        matrices_list = []
        temporal_matrices_list = []
        distance_matrices_list = []
        duration_matrices_list = []
        # week
        matrices_week_list = []
        temporal_matrices_week_list = []
        distance_matrices_week_list = []
        duration_matrices_week_list = []
        # weekend
        matrices_weekend_list = []
        temporal_matrices_weekend_list = []
        distance_matrices_weekend_list = []
        duration_matrices_weekend_list = []

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
            # week
            matrix_week_df = inputs['week']['adjacency']['matrices'].tolist()
            temporal_week_df = inputs['week']['temporal']['matrices'].tolist()
            distance_week_df = inputs['week']['distance']['matrices'].tolist()
            duration_week_df = inputs['week']['duration']['matrices'].tolist()
            # weekend
            matrix_weekend_df = inputs['weekend']['adjacency']['matrices'].tolist()
            temporal_weekend_df = inputs['weekend']['temporal']['matrices'].tolist()
            distance_weekend_df = inputs['weekend']['distance']['matrices'].tolist()
            duration_weekend_df = inputs['weekend']['duration']['matrices'].tolist()


        if len(ids) != len(matrix_df):
            print("ERRO TAMANHO DA MATRIZ")
            exit()

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
            if user_matrix.shape[0] < max_size_matrices:
                remove_users_ids.append(user_id)
                continue
            size = user_matrix.shape[0]
            if size > maior:
                maior = size

            # matrices get new size, equal for everyone
            if model_name == "poi_gnn":
                if week and weekend:
                    user_matrix, user_matrix_week, user_matrix_weekend, user_category, idx = self._resize_adjacency_and_category_matrices(user_matrix, user_matrix_week, user_matrix_weekend, user_category, max_size_matrices)
                    #print("user matrix week: ", user_matrix_week)
                else:
                    user_matrix, user_category, idx = self._resize_adjacency_and_category_matrices_baselines(
                        user_matrix, user_category, max_size_matrices)
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

            """feature"""
            user_temporal_matrix = temporal_df[i]
            user_temporal_matrix = json.loads(user_temporal_matrix)
            user_temporal_matrix = np.array(user_temporal_matrix)
            user_temporal_matrix = user_temporal_matrix[idx]
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


            if model_name == "poi_gnn":
                """distance"""
                user_distance_matrix = distance_df[i]
                user_distance_matrix = json.loads(user_distance_matrix)
                user_distance_matrix = np.array(user_distance_matrix)
                user_distance_matrix = user_distance_matrix[idx[:,None], idx]
                # week
                user_distance_matrix_week = distance_week_df[i]
                user_distance_matrix_week = json.loads(user_distance_matrix_week)
                user_distance_matrix_week = np.array(user_distance_matrix_week)
                user_distance_matrix_week = user_distance_matrix_week[idx[:,None], idx]
                # weekend
                user_distance_matrix_weekend = distance_weekend_df[i]
                user_distance_matrix_weekend = json.loads(user_distance_matrix_weekend)
                user_distance_matrix_weekend = np.array(user_distance_matrix_weekend)
                user_distance_matrix_weekend = user_distance_matrix_weekend[idx[:,None], idx]

                """duration"""
                user_duration_matrix = duration_df[i]
                user_duration_matrix = json.loads(user_duration_matrix)
                user_duration_matrix = np.array(user_duration_matrix)
                user_duration_matrix = user_duration_matrix[idx[:,None], idx]
                # week
                user_duration_matrix_week = duration_week_df[i]
                user_duration_matrix_week = json.loads(user_duration_matrix_week)
                user_duration_matrix_week = np.array(user_duration_matrix_week)
                user_duration_matrix_week = user_duration_matrix_week[idx[:,None], idx]
                # weekend
                user_duration_matrix_weekend = duration_weekend_df[i]
                user_duration_matrix_weekend = json.loads(user_duration_matrix_weekend)
                user_duration_matrix_weekend = np.array(user_duration_matrix_weekend)
                user_duration_matrix_weekend = user_duration_matrix_weekend[idx[:,None], idx]

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
                distance_matrices_week_list.append(user_distance_matrix_week)
                duration_matrices_week_list.append(user_duration_matrix_week)
                # weekend
                matrices_weekend_list.append(user_matrix_weekend)
                temporal_matrices_weekend_list.append(user_temporal_matrix_weekend)
                distance_matrices_weekend_list.append(user_distance_matrix_weekend)
                duration_matrices_weekend_list.append(user_duration_matrix_weekend)


        self.features_num_columns = temporal_matrices_list[-1].shape[1]
        matrices_list = np.array(matrices_list)
        temporal_matrices_list = np.array(temporal_matrices_list)
        users_categories = np.array(users_categories)
        if model_name == "poi_gnn":
            distance_matrices_list = np.array(distance_matrices_list)
            duration_matrices_list = np.array(duration_matrices_list)

            # week
            matrices_week_list = np.array(matrices_week_list)
            temporal_matrices_week_list = np.array(temporal_matrices_week_list)
            distance_matrices_week_list = np.array(distance_matrices_week_list)
            duration_matrices_week_list = np.array(duration_matrices_weekend_list)

            # weekend
            matrices_weekend_list = np.array(matrices_weekend_list)
            temporal_matrices_weekend_list = np.array(temporal_matrices_weekend_list)
            temporal_matrices_week_list = np.array(temporal_matrices_week_list)
            distance_matrices_weekend_list = np.array(distance_matrices_weekend_list)
            duration_matrices_weekend_list = np.array(duration_matrices_weekend_list)
        print("antes", matrices_list.shape, temporal_matrices_list.shape)

        if model_name == "poi_gnn":
            if week and weekend:
                return (users_categories, matrices_list, temporal_matrices_list, distance_matrices_list, duration_matrices_list,
                        matrices_week_list, temporal_matrices_week_list, distance_matrices_week_list,
                        duration_matrices_week_list, matrices_weekend_list, temporal_matrices_weekend_list,
                        distance_matrices_weekend_list, duration_matrices_weekend_list)
            else:
                return (matrices_list, users_categories, temporal_matrices_list,distance_matrices_list,
                        duration_matrices_list, remove_users_ids)
        else:
            return (matrices_list, users_categories, temporal_matrices_list)

    def generate_nodes_ids(self, rows, cols):


        ids = []
        for i in range(rows):
            row = [i for i in range(cols)]
            ids.append(row)

        return np.array(ids)

    def k_fold_split_train_test(self,
                                inputs,
                                n_splits,
                                week_type,
                                model_name='poi_gnn'):

        adjacency_list = inputs[week_type]['adjacency']
        temporal_list = inputs[week_type]['temporal']
        user_categories = inputs[week_type]['categories']
        if model_name == "poi_gnn":
            distance_list = inputs[week_type]['distance']
            duration_list = inputs[week_type]['duration']
        else:
            distance_list = []
            duration_list = []
        skip = False
        if n_splits == 1:
            skip = True
            n_splits = 2
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=0)

        folds = []
        classes_weights = []
        for train_indexes, test_indexes in kf.split(adjacency_list):

            fold, class_weight = self._split_train_test(adjacency_list,
                                                        user_categories,
                                                        temporal_list,
                                                        distance_list,
                                                        duration_list,
                                                        train_indexes,
                                                        test_indexes)
            folds.append(fold)
            classes_weights.append(class_weight)
            if skip:
                break

        return folds, classes_weights

    def _split_train_test(self,
                          adjacency_list,
                          user_categories,
                          temporal_list,
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
        else:
            distance_list_train = []
            duration_list_train = []


        adjacency_list_test = adjacency_list[test_indexes]
        user_categories_test = user_categories[test_indexes]
        temporal_list_test = temporal_list[test_indexes]
        if len(distance_list) > 0:
            distance_list_test = distance_list[test_indexes]
            duration_list_test = duration_list[test_indexes]
        else:
            distance_list_test = []
            duration_list_test = []

        flatten_train_category = user_categories_train.flatten()
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
                    adjacency_list_test, user_categories_test, temporal_list_test, distance_list_test,
                    duration_list_test), class_weight
        else:
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
                                                         country,
                                                         version):

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

                history, report, model, accuracy = self.train_and_evaluate_model(fold,
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
                                 version="normal",
                                 model=None):


        #print("entradas: ", adjacency_train.shape, features_train.shape, y_train.shape)
        #print("enstrada test: ", adjacency_test.shape, features_test.shape, y_test.shape)
        adjacency_train, y_train, temporal_train, distance_train, duration_train, \
        adjacency_test, y_test, temporal_test, distance_test, duration_test = fold
        adjacency_week_train, y_train_week, temporal_train_week, distance_week_train, duration_week_train, \
        adjacency_test_week, y_test_week, temporal_test_week, distance_week_test, \
        duration_week_test = fold_week
        adjacency_train_weekend, y_train_weekend, temporal_train_weekend, distance_weekend_train, \
        duration_weekend_train, adjacency_test_weekend, y_test_weekend, temporal_test_weekend, distance_weekend_test, \
        duration_weekend_test = fold_weekend

        print("Tamanho das matrizes de treino: ", adjacency_train.shape, temporal_train.shape,
              adjacency_week_train.shape, temporal_train_week.shape)

        print("Tamanho das matrizes de teste: ", adjacency_test.shape, temporal_test.shape,
              adjacency_test_week.shape, temporal_test_week.shape)
        # verifying whether categories arrays are equal
        compare1 = y_train == y_train_week
        compare2 = y_train_week == y_train_weekend
        compare3 = y_test == y_test_week
        compare4 = y_test_week == y_test_weekend
        if not(compare1.all() and compare2.all() and compare3.all() and compare4.all()):
            print("Listas difernetes de categorias")
            exit()

        num_classes = max(y_train.flatten()) + 1
        max_size = max_size_matrices
        print("Quantidade de classes: ", num_classes)
        if country == 'BR':
            if version == "normal":
                print("Tipo de rede neural: NORMAL")
                model = GNN(num_classes, max_size, max_size_sequence,
                            self.features_num_columns).build(seed=seed)
            elif version == "PATH":
                print("PATH")
                model = GNNPath(num_classes, max_size, max_size_sequence,
                            self.features_num_columns).build(seed=seed)
        elif country == 'US':
            model = GNNUS(num_classes, max_size, max_size_sequence,
                        self.features_num_columns).build(seed=seed)
        if country == 'US':
            batch = max_size * 6
        elif country == 'BR':
            batch = max_size * 5

        print("Tamanho do batch: ", batch)
        model.compile(optimizer="adam", loss=['categorical_crossentropy'],
                      weighted_metrics=[tf.keras.metrics.CategoricalAccuracy(name="acc")
                                        ])
        y_train = np_utils.to_categorical(y_train, num_classes=num_classes)
        y_test = np_utils.to_categorical(y_test, num_classes=num_classes)
        hi = model.fit(x=[adjacency_train, temporal_train, distance_train, duration_train],
                       y=y_train, validation_data=([adjacency_test, temporal_test, distance_test, duration_test], y_test),
                       epochs=epochs, batch_size=batch,
                       shuffle=False,  # Shuffling data means shuffling the whole graph
                       callbacks=[
                           EarlyStopping(patience=100, restore_best_weights=True)
                       ]
                       )

        h = hi.history
        #print("summary: ", model.summary())
        y_predict_location = model.predict([adjacency_test, temporal_test,distance_test, duration_test],
                                           batch_size=batch)

        scores = model.evaluate([adjacency_test, temporal_test, distance_test, duration_test],
                                y_test, batch_size=batch)
        print("scores: ", scores)

        # To transform one_hot_encoding to list of integers, representing the locations
        y_predict_location = one_hot_decoding_predicted(y_predict_location)
        y_test = one_hot_decoding_predicted(y_test)
        report = skm.classification_report(y_test, y_predict_location, output_dict=True)
        # print(report)

        return h, report, model, report['accuracy']

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


        