import numpy as np
import json
import pandas as pd
import sklearn.preprocessing
import sklearn as skm
import spektral as sk
import tensorflow as tf
from sklearn.model_selection import KFold
from tensorflow.keras import utils as np_utils

from utils.nn_preprocessing import one_hot_decoding_predicted
from domain.poi_categorization_domain import PoiCategorizationDomain
from model.neural_network.poi_categorization_baselines.BR.gpr.model import GPRModel
from model.neural_network.poi_categorization_baselines.US.gpr.model import GPRUSModel
from utils.nn_preprocessing import one_hot_decoding_predicted, top_k_rows



class PoiCategorizationBaselineGPRDomain(PoiCategorizationDomain):


    def __init__(self, dataset_name):
        super().__init__(dataset_name)

    def adjacency_preprocessing(self,
                                matrix_df,
                                feature_df,
                                user_poi_vector_df,
                                max_size_matrices,
                                categories_type,
                                model_name="gcn"):

        matrices_list = []
        distance_matrices_list = []
        user_poi_vector_list = []

        users_categories = []
        flatten_users_categories = []
        maior = -10
        remove_users_ids = []

        ids = matrix_df['user_id'].unique().tolist()

        for i in range(matrix_df.shape[0]):
            user_id = matrix_df['user_id'].iloc[i]
            user_matrix = matrix_df['matrices'].iloc[i]
            user_category = matrix_df['category'].iloc[i]
            user_matrix = json.loads(user_matrix)
            user_matrix = np.array(user_matrix)
            user_category = json.loads(user_category)
            user_category = np.array(user_category)
            if user_matrix.shape[0] < max_size_matrices:
                remove_users_ids.append(user_id)
                continue
            size = user_matrix.shape[0]
            if size > maior:
                maior = size

            # matrices get new size, equal for everyone

            user_matrix, user_category, idx = self._resize_adjacency_and_category_matrices_gpr(user_matrix,
                                                                                           user_category,
                                                                                           max_size_matrices,
                                                                                           categories_type)



            # feature
            user_distance_matrix = feature_df.iloc[i]
            user_distance_matrix = user_distance_matrix['matrices']
            user_distance_matrix = json.loads(user_distance_matrix)
            user_distance_matrix = np.array(user_distance_matrix)
            user_distance_matrix = user_distance_matrix[idx[:,None], idx]

            # converter metros para km
            # user_distance_matrix = user_distance_matrix/1000
            # user_distance_matrix = np.where(user_distance_matrix>100, 100, user_distance_matrix)
            # user_distance_matrix = skm.preprocessing.normalize(user_distance_matrix)

            # sequence
            user_poi_vector = user_poi_vector_df.iloc[i]
            user_poi_vector = user_poi_vector['matrices']
            user_poi_vector = json.loads(user_poi_vector)
            user_poi_vector = np.array(user_poi_vector)
            user_poi_vector = np.divide(user_poi_vector, np.max(user_poi_vector))
            #user_poi_vector = user_poi_vector[idx]
            user_poi_vector_list.append(user_poi_vector.tolist())

            matrices_list.append(user_matrix)
            users_categories.append(user_category)
            flatten_users_categories = flatten_users_categories + user_category.tolist()
            distance_matrices_list.append(user_distance_matrix)


        self.features_num_columns = distance_matrices_list[-1].shape[0]
        matrices_list = np.array(matrices_list)
        distance_matrices_list = np.array(distance_matrices_list)
        user_poi_vector_list = np.array(user_poi_vector_list)
        users_categories = np.array(users_categories)

        print("antes", matrices_list.shape, distance_matrices_list.shape, user_poi_vector_list.shape)

        return matrices_list, users_categories, distance_matrices_list, user_poi_vector_list, remove_users_ids

    def _resize_adjacency_and_category_matrices_gpr(self, user_matrix, user_category, max_size_matrices, dataset_name):

        k = max_size_matrices
        if user_matrix.shape[0] < k:
            k = user_matrix.shape[0]
        # select the k rows that have the highest sum
        idx = top_k_rows(user_matrix, k)
        user_category = user_category[idx]
        user_matrix = user_matrix[idx[:, None], idx]

        return user_matrix, user_category, idx

    def preprocess_adjacency_matrix_by_gnn_type(self, matrices, model_name):

        new_matrices = []
        if model_name == "gcn" or model_name == "gae":
            for i in range(len(matrices)):
                new_matrices.append(sk.layers.GraphConv.preprocess(matrices[i]))
        elif model_name == "arma" or model_name == "arma_enhanced":
            for i in range(len(matrices)):
                new_matrices.append(sk.layers.ARMAConv.preprocess(matrices[i]))
        elif model_name == "diff":
            for i in range(len(matrices)):
                new_matrices.append(sk.layers.DiffusionConv.preprocess(matrices[i]))


        return  np.array(new_matrices)

    def k_fold_split_train_test_gpr(self,
                                n_splits,
                                inputs,
                                model_name='gpr'):

        adjacency_list = inputs['adjacency']
        distance_list = inputs['distance']
        user_poi_list = inputs['user_poi']
        categories_list = inputs['categories']
        if n_splits == 1:
            skip = True
            n_splits = 2
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=0)

        folds = []
        classes_weights = []
        for train_indexes, test_indexes in kf.split(adjacency_list):

            fold = self._split_train_test_gpr(n_splits,
                                                        model_name,
                                                        adjacency_list,
                                                        user_poi_list,
                                                        distance_list,
                                                        categories_list,
                                                        train_indexes,
                                                        test_indexes)
            folds.append(fold)

        return folds

    def _split_train_test_gpr(self,
                          k,
                          model_name,
                          adjacency_list,
                          user_poi_list,
                          distance_list,
                          user_categories,
                          train_indexes,
                          test_indexes):

        size = adjacency_list.shape[0]
        # 'average', 'cv', 'median', 'radius', 'label'
        adjacency_list_train = adjacency_list[train_indexes]
        adjacency_list_test = adjacency_list[test_indexes]
        user_categories_train = user_categories[train_indexes]
        user_categories_test = user_categories[test_indexes]

        distance_list_train = distance_list[train_indexes]
        distance_list_test = distance_list[test_indexes]
        user_poi_train = user_poi_list[train_indexes]
        user_poi_test = user_poi_list[test_indexes]


        user_categories_train = np.array([[e for e in row] for row in user_categories_train])
        user_categories_test = np.array([[e for e in row] for row in user_categories_test])

        return (adjacency_list_train, user_categories_train, distance_list_train, user_poi_train, adjacency_list_test, user_categories_test, distance_list_test, user_poi_test)



    def k_fold_with_replication_train_and_evaluate_baselines_model(self,
                                                         folds,
                                                         n_replications,
                                                         class_weight,
                                                         max_size_matrices,
                                                         base_report,
                                                        parameters,
                                                        country):

        folds_histories = []
        folds_reports = []
        iteration = 0
        for i in range(len(folds)):

            fold = folds[i]
            histories = []
            reports = []

            for j in range(n_replications):

                history, report = self.train_and_evaluate_baseline_model(fold,
                                                        max_size_matrices,
                                                        parameters,
                                                        class_weight,
                                                        country,
                                                        seed=iteration)
                iteration+=1

                base_report = self._add_location_report(base_report, report)
                histories.append(history)
                #reports.append(report)
            folds_histories.append(histories)
            #folds_reports.append(reports)

        return folds_histories, base_report

    def train_and_evaluate_baseline_model(self,
                                          fold,
                                          max_size_matrices,
                                          parameters,
                                          class_weight,
                                          country,
                                          seed=None):

        adjacency_train, y_train, distance_train, user_poi_train, adjacency_test, y_test, distance_test, user_poi_test = fold
        class_weight = list(class_weight.values())
        new_class_weight_train = []

        for i in range(len(user_poi_train)):

            new_class_weight_train.append([])
            for j in range(max_size_matrices):

                new_class_weight_train[i].append(user_poi_train[i])

        new_class_weight_test = []

        for i in range(len(user_poi_test)):

            new_class_weight_test.append([])
            for j in range(max_size_matrices):
                new_class_weight_test[i].append(user_poi_test[i])

        new_class_weight_train = np.array(new_class_weight_train)
        new_class_weight_test = np.array(new_class_weight_test)

        y_train = np_utils.to_categorical(y_train, num_classes=7)
        y_test = np_utils.to_categorical(y_test, num_classes=7)
        print("entradas: ", adjacency_train.shape, distance_train.shape, user_poi_train.shape, new_class_weight_train.shape,
              y_train.shape)
        print("enstrada test: ", adjacency_test.shape, distance_test.shape, user_poi_test.shape, new_class_weight_test.shape,
              y_test.shape)

        transposed_adjacency_train = self.transpose_matrices(adjacency_train)
        transposed_adjacency_test = self.transpose_matrices(adjacency_test)


        max_size = max_size_matrices
        print("classes: ", 7)
        batch = max_size*30
        print("tamanho batch: ", batch)
        print("epocas: ", parameters['epochs'])
        print("y_train: ", y_train.shape, y_test.shape)

        model = GPRUSModel(classes=7, max_size_matrices=max_size,
                             features_num_columns=self.features_num_columns).build(output_size=7, seed=seed)
        model.compile(optimizer=parameters['optimizer'],
                      loss=[parameters['loss'], tf.keras.losses.MeanSquaredError()],
                      weighted_metrics=[tf.keras.metrics.CategoricalAccuracy(name="acc"),
                                        tf.keras.metrics.CategoricalAccuracy(name="acc")])


        hi = model.fit(x=[transposed_adjacency_train, adjacency_train, distance_train, user_poi_train, new_class_weight_train],
                       y=y_train, validation_data=([transposed_adjacency_test, adjacency_test, distance_test, user_poi_test, new_class_weight_test], y_test),
                       epochs=parameters['epochs'], batch_size=batch)

        h = hi.history
        # h = {'loss': h['loss'], 'val_loss': h['val_loss']}
        #print("summary: ", model.summary())

        y_predict_location = model.predict([transposed_adjacency_test, adjacency_test, distance_test, user_poi_test, new_class_weight_test],
                                           batch_size=batch)

        scores = model.evaluate([transposed_adjacency_test, adjacency_test, distance_test, user_poi_test, new_class_weight_test],
                                y_test, batch_size=batch)

        # To transform one_hot_encoding to list of integers, representing the locations
        # print("------------- Location ------------")
        # print("saida: ", y_predict_location[0].shape, y_predict_location, y_predict_location.shape)
        y_predict_location = one_hot_decoding_predicted(y_predict_location)
        y_test = one_hot_decoding_predicted(y_test)
        # print("Original: ", y_test[0], " tamanho: ", len(y_test))
        # print("previu: ", y_predict_location[0], " tamanho: ", len(y_predict_location))
        report = skm.metrics.classification_report(y_test, y_predict_location, output_dict=True)
        print(report)
        return h, report

    def transpose_matrices(self, matrices):

        transposed_matrices = []

        for m in matrices:
            transposed_matrices.append(m.T)

        return np.array(transposed_matrices)




        