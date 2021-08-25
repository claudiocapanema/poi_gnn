import pandas as pd
import numpy as np
import json
from sklearn.preprocessing import normalize
from sklearn.model_selection import KFold

import spektral as sk
import tensorflow as tf
import sklearn.metrics as skm
from tensorflow.keras import utils as np_utils
from functools import partial

from loader.file_loader import FileLoader
from extractor.file_extractor import FileExtractor
from model.neural_network.gnn_transfer_learning import GNNTransferLearning
from utils.nn_preprocessing import one_hot_decoding_predicted, top_k_rows, weighted_categorical_crossentropy


class PoiCategorizationTransferLearningDomain:


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

    def read_matrix(self, adjacency_matrix_filename, feature_matrix_filename):

        adjacency_df = self.file_extractor.read_csv(adjacency_matrix_filename).drop_duplicates(subset=['user_id'])
        feature_df = self.file_extractor.read_csv(feature_matrix_filename).drop_duplicates(subset=['user_id'])
        if adjacency_df['user_id'].tolist() != feature_df['user_id'].tolist():
            print("MATRIZES DIFERENTES")

        return adjacency_df, feature_df

    def read_matrices(self, adjacency_matrix_dir, feature_matrix_dir):

        adjacency_df = self.file_extractor.read_multiples_csv(adjacency_matrix_dir)
        feature_df = self.file_extractor.read_multiples_csv(feature_matrix_dir)

        return adjacency_df, feature_df

    def read_users_metrics(self, filename):

        return self.file_extractor.read_csv(filename).drop_duplicates(subset=['user_id'])

    def _resize_adjacency_and_category_matrices(self, user_matrix, user_category, max_size_matrices, categories_type):

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


    def adjacency_preprocessing(self,
                                matrix_df,
                                feature_df,
                                users_metrics_ids,
                                osm_categories,
                                max_size_matrices,
                                categories_type,
                                model_name="gcn"):

        matrices_list = []
        features_matrices_list = []

        users_categories = []
        flatten_users_categories = []
        maior = -10
        remove_users_ids = []

        for i in range(matrix_df.shape[0]):
            user_id = matrix_df['user_id'].iloc[i]
            if user_id not in users_metrics_ids:
                print("diferentes", user_id)
                remove_users_ids.append(user_id)
                continue
            user_matrix = matrix_df['matrices'].iloc[i]
            # user_category = first_level_categories[i]
            user_category = matrix_df['category'].iloc[i]
            user_matrix = json.loads(user_matrix)
            user_matrix = np.array(user_matrix)
            user_category = json.loads(user_category)
            user_category = np.array(user_category)
            if user_matrix.shape[0]<max_size_matrices:
                remove_users_ids.append(user_id)
                continue
            size = user_matrix.shape[0]
            if size > maior:
                maior = size

            # matrices get new size, equal for everyone

            user_matrix, user_category, idx = self._resize_adjacency_and_category_matrices(user_matrix, user_category, max_size_matrices, categories_type)

            if model_name == "gcn" or model_name == "gae":
                user_matrix = sk.layers.GraphConv.preprocess(user_matrix)
            elif model_name == "arma":
                user_matrix = sk.layers.ARMAConv.preprocess(user_matrix)
            matrices_list.append(user_matrix)
            users_categories.append(user_category)
            flatten_users_categories = flatten_users_categories + user_category.tolist()

            # feature

            user_feature_matrix = feature_df['matrices'].iloc[i]
            user_feature_matrix = json.loads(user_feature_matrix)
            user_feature_matrix = np.array(user_feature_matrix)
            # user_matrix = self._preprocess_features(user_matrix)
            user_feature_matrix = self._resize_features_matrix(user_feature_matrix, idx, max_size_matrices)
            features_matrices_list.append(user_feature_matrix)

        self.features_num_columns = features_matrices_list[-1].shape[1]
        matrices_list = np.array(matrices_list)
        features_matrices_list = np.array(features_matrices_list)
        users_categories = np.array(users_categories)

        return matrices_list, users_categories, features_matrices_list, remove_users_ids

    def feature_preprocessing(self, matrix_df):

        matrices_list = []
        matrix_df['category'] = self._first_level_categories(matrix_df['category'])
        for i in range(matrix_df.shape[0]):
            user_feature_matrix = matrix_df['matrices'].iloc[i]
            user_category = matrix_df['category'].iloc[i]
            user_category = np.array(user_category)
            user_feature_matrix = json.loads(user_feature_matrix)
            user_feature_matrix = np.array(user_feature_matrix)
            #user_matrix = self._preprocess_features(user_matrix)
            user_feature_matrix, user_category = self._resize_features_matrix(user_feature_matrix, user_category)
            matrices_list.append(user_feature_matrix)

        matrices_list = np.array(matrices_list)
        return  matrices_list

    def generate_nodes_ids(self, rows, cols):


        ids = []
        for i in range(rows):
            row = [i for i in range(cols)]
            ids.append(row)

        return np.array(ids)

    def k_fold_split_train_test(self,
                                adjacency_list,
                                user_categories,
                                features_list,
                                train_size,
                                n_splits,
                                users_metrics=None):

        kf = KFold(n_splits=n_splits)

        folds = []
        for train_indexes, test_indexes in kf.split(adjacency_list):

            fold, class_weight = self._split_train_test(adjacency_list,
                                  user_categories,
                                  features_list,
                                  train_size,
                                    users_metrics,
                                  train_indexes,
                                  test_indexes)
            folds.append(fold)

        return folds, class_weight

    def _split_train_test(self,
                         adjacency_list,
                         user_categories,
                         features_list,
                         train_size,
                          users_metrics=None,
                         train_indexes=None,
                         test_indexes=None):

        size = adjacency_list.shape[0]
        if users_metrics is not None:
            users_metrics = users_metrics[['average']].to_numpy()
        # 'average', 'cv', 'median', 'radius', 'label'
        if train_indexes is None or test_indexes is None:
            train_index = int(size*train_size)
            train_indexes = [i for i in range(0, train_index)]
            test_indexes = [i for i in range(train_index, size)]
        adjacency_list_train = adjacency_list[train_indexes]
        user_categories_train = user_categories[train_indexes]
        features_list_train = features_list[train_indexes]
        if users_metrics is not None:
            user_metrics_list_train = users_metrics[train_indexes]
        else:
            user_metrics_list_train = np.ones(shape=1)

        adjacency_list_test = adjacency_list[test_indexes]
        user_categories_test = user_categories[test_indexes]
        features_list_test = features_list[test_indexes]
        if users_metrics is not None:
            user_metrics_list_test = users_metrics[test_indexes]
        else:
            user_metrics_list_test = np.ones(shape=1)

        flatten_train_category = user_categories_train.flatten()
        flatten_train_category = pd.Series(flatten_train_category, name='category')
        flatten_train_category = flatten_train_category[flatten_train_category>0]
        flatten_train_category = flatten_train_category.astype('object')
        train_categories_freq = {e:0 for e in flatten_train_category.unique().tolist()}
        for i in range(flatten_train_category.shape[0]):
            train_categories_freq[flatten_train_category.iloc[i]]+=1
        n = sum(train_categories_freq.values())

        for e in train_categories_freq:
            train_categories_freq[e] = train_categories_freq[e]/n
        train_categories_freq[0] = 0
        class_weight = list(train_categories_freq.values())
        user_categories_train = np.array([[e for e in row] for row in user_categories_train])
        user_categories_test = np.array([[e for e in row] for row in user_categories_test])
        print("forma: ", adjacency_list_train.shape, user_metrics_list_train.shape, adjacency_list_test.shape, user_metrics_list_test.shape)
        return (adjacency_list_train, user_categories_train, features_list_train, user_metrics_list_train, \
               adjacency_list_test, user_categories_test, features_list_test, user_metrics_list_test), \
               train_categories_freq


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
                                                         base_model,
                                                         folds,
                                                         n_replications,
                                                         class_weight,
                                                         categories_to_int_osm,
                                                         max_size_matrices,
                                                         base_report):

        folds_histories = []
        folds_reports = []
        models = []
        accuracies = []
        for fold in folds:
            histories = []
            reports = []
            adjacency_train, y_train, features_train, user_metrics_train, \
            adjacency_test, y_test, features_test, user_metrics_test = fold
            for i in range(n_replications):

                history, report, model, accuracy = self.train_and_evaluate_model(base_model,
                                                                                 adjacency_train,
                                                        features_train,
                                                        y_train,
                                                        user_metrics_train,
                                                        adjacency_test,
                                                        features_test,
                                                        y_test,
                                                        user_metrics_test,
                                                        class_weight,
                                                        categories_to_int_osm,
                                                        max_size_matrices)

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
                                 base_model,
                                 adjacency_train,
                                 features_train,
                                 y_train,
                                 user_metrics_train,
                                 adjacency_test,
                                 features_test,
                                 y_test,
                                 user_metrics_test,
                                 class_weight,
                                 categories_to_int_osm,
                                 max_size_matrices,
                                 model=None):


        #print("entradas: ", adjacency_train.shape, features_train.shape, y_train.shape)
        #print("enstrada test: ", adjacency_test.shape, features_test.shape, y_test.shape)
        num_classes = len(pd.Series(list(categories_to_int_osm.values())).unique().tolist())
        max_size = max_size_matrices
        print("classes: ", num_classes, adjacency_train.shape)
        model = GNNTransferLearning(num_classes,
                                    max_size,
                                    base_model,
                    self.features_num_columns).build()
        batch = max_size*10
        print("y_train: ", y_train.shape, y_test.shape)
        w1 = np.array([class_weight.values()]*max_size)
        loss1 = partial(weighted_categorical_crossentropy, weights=w1)
        model.compile(optimizer="adam", loss=['categorical_crossentropy'],
                      weighted_metrics=[tf.keras.metrics.Accuracy(name="acc"),
                                        tf.keras.metrics.Precision(name="Precision"),
                                        tf.keras.metrics.Recall(name="Recall")])
        y_train = np_utils.to_categorical(y_train, num_classes=num_classes)
        y_test = np_utils.to_categorical(y_test, num_classes=num_classes)

        hi = model.fit(x=[adjacency_train, features_train, user_metrics_train],
                       y=y_train, validation_data=([adjacency_test, features_test, user_metrics_test], y_test),
                       epochs=15, batch_size=batch)

        h = hi.history
        #print("summary: ", model.summary())

        y_predict_location = model.predict([adjacency_test, features_test, user_metrics_test],
                                           batch_size=batch)

        scores = model.evaluate([adjacency_test, features_test, user_metrics_test],
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

    def read_model(self, filename):

        model = self.file_extractor.read_model(filename)

        return model






        