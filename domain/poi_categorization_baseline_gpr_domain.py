import numpy as np
import json
import spektral as sk
import tensorflow as tf
import sklearn.metrics as skm
from tensorflow.keras import utils as np_utils

from utils.nn_preprocessing import one_hot_decoding_predicted
from domain.poi_categorization_domain import PoiCategorizationDomain
from model.neural_network.poi_categorization_baselines.BR.gpr.model import GPRModel
from model.neural_network.poi_categorization_baselines.US.gpr.model import GPRUSModel



class PoiCategorizationBaselineGPRDomain(PoiCategorizationDomain):


    def __init__(self, dataset_name):
        super().__init__(dataset_name)

    def adjacency_preprocessing(self,
                                matrix_df,
                                feature_df,
                                user_poi_vector_df,
                                users_metrics_ids,
                                max_size_matrices,
                                max_size_sequence,
                                categories_type,
                                model_name="gcn"):

        matrices_list = []
        features_matrices_list = []
        user_poi_vector_list = []

        users_categories = []
        flatten_users_categories = []
        maior = -10
        remove_users_ids = []

        ids = matrix_df['user_id'].unique().tolist()

        for i in range(matrix_df.shape[0]):
            user_id = matrix_df['user_id'].iloc[i]
            if user_id not in users_metrics_ids:
                print("diferentes", user_id)
                remove_users_ids.append(user_id)
                continue
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

            user_matrix, user_category, idx = self._resize_adjacency_and_category_matrices(user_matrix,
                                                                                           user_category,
                                                                                           max_size_matrices,
                                                                                           categories_type)



            # feature
            user_feature_matrix = feature_df.iloc[i]
            user_feature_matrix = user_feature_matrix['matrices']
            user_feature_matrix = json.loads(user_feature_matrix)
            user_feature_matrix = np.array(user_feature_matrix)
            user_feature_matrix = user_feature_matrix[idx[:,None], idx]
            # converter metros para km
            user_feature_matrix = user_feature_matrix/1000
            user_feature_matrix = np.where(user_feature_matrix>100, 100, user_feature_matrix)

            # sequence
            if user_poi_vector_df is not None:
                user_poi_vector = user_poi_vector_df.iloc[i]
                user_poi_vector = user_poi_vector['vector']
                user_poi_vector = json.loads(user_poi_vector)
                user_poi_vector = np.array(user_poi_vector)
                user_poi_vector = user_poi_vector[idx]
                user_poi_vector_list.append(user_poi_vector.tolist())

            matrices_list.append(user_matrix)
            users_categories.append(user_category)
            flatten_users_categories = flatten_users_categories + user_category.tolist()
            features_matrices_list.append(user_feature_matrix)


        self.features_num_columns = features_matrices_list[-1].shape[0]
        matrices_list = np.array(matrices_list)
        features_matrices_list = np.array(features_matrices_list)
        user_poi_vector_list = np.array(user_poi_vector_list)
        users_categories = np.array(users_categories)

        print("antes", matrices_list.shape, features_matrices_list.shape, user_poi_vector_list.shape)

        return matrices_list, users_categories, features_matrices_list, user_poi_vector_list, remove_users_ids

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

    def k_fold_with_replication_train_and_evaluate_baselines_model(self,
                                                         folds,
                                                         n_replications,
                                                         classes_weights,
                                                         max_size_matrices,
                                                        max_size_sequence,
                                                         base_report,
                                                        parameters,
                                                        augmentation_categories,
                                                        country):

        folds_histories = []
        folds_reports = []
        iteration = 0
        for i in range(len(folds)):

            fold = folds[i]
            class_weight = classes_weights[i]
            histories = []
            reports = []
            adjacency_train, y_train, features_train, sequence_train, user_metrics_train, \
            adjacency_test, y_test, features_test, sequence_test, user_metrics_test = fold

            print("antes: ", adjacency_train.shape, y_train.shape, features_train.shape, user_metrics_train.shape)
            # adjacency_train, features_train, y_train, user_metrics_train = self._augmentate_training_data(
            #     adjacency_matrices=adjacency_train, features_matrices=features_train,
            #     categories=y_train, user_metrics=None,
            #     augmentation_cateogories=augmentation_categories)

            # adjacency_train = self.preprocess_adjacency_matrix_by_gnn_type(adjacency_train, model_name)
            # adjacency_test = self.preprocess_adjacency_matrix_by_gnn_type(adjacency_test, model_name)

            # permutation_indices = np.random.permutation(len(y_train))
            # print("reorden", permutation_indices)
            # adjacency_train = adjacency_train[permutation_indices]
            # features_train = features_train[permutation_indices]
            # y_train = y_train[permutation_indices]
            print("depois: ", adjacency_train.shape, y_train.shape, features_train.shape, user_metrics_train.shape)
            for j in range(n_replications):

                history, report = self.train_and_evaluate_baseline_model(adjacency_train,
                                                        features_train,
                                                        sequence_train,
                                                        y_train,
                                                        adjacency_test,
                                                        features_test,
                                                        sequence_test,
                                                        y_test,
                                                        max_size_matrices,
                                                        max_size_sequence,
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
                                          adjacency_train,
                                          features_train,
                                          sequence_train,
                                          y_train,
                                          adjacency_test,
                                          features_test,
                                          sequence_test,
                                          y_test,
                                          max_size_matrices,
                                          max_size_sequence,
                                          parameters,
                                          class_weight,
                                          country,
                                          seed=None):


        print("entradas: ", adjacency_train.shape, features_train.shape, sequence_train.shape,
              y_train.shape)
        print("enstrada test: ", adjacency_test.shape, features_test.shape, sequence_test.shape,
              y_test.shape)
        transposed_adjacency_train = self.transpose_matrices(adjacency_train)
        transposed_adjacency_test = self.transpose_matrices(adjacency_test)
        sequence_train = self.transpose_matrices(sequence_train)
        sequence_test = self.transpose_matrices(sequence_test)

        num_classes = max(y_train.flatten()) + 1
        max_size = max_size_matrices
        print("classes: ", num_classes, adjacency_train.shape)
        batch = max_size*30
        print("tamanho batch: ", batch)
        print("epocas: ", parameters['epochs'])
        print("y_train: ", y_train.shape, y_test.shape)
        if country == 'BR':
            model = GPRModel(classes=num_classes, max_size_matrices=max_size, features_num_columns=self.features_num_columns).build(output_size=num_classes, seed=seed)
        elif country == 'US':
            model = GPRUSModel(classes=num_classes, max_size_matrices=max_size,
                             features_num_columns=self.features_num_columns).build(output_size=num_classes, seed=seed)
        y_train = np_utils.to_categorical(y_train, num_classes=num_classes)
        y_test = np_utils.to_categorical(y_test, num_classes=num_classes)
        model.compile(optimizer=parameters['optimizer'],
                      loss=[parameters['loss'], tf.keras.losses.MeanSquaredError()],
                      weighted_metrics=[tf.keras.metrics.CategoricalAccuracy(name="acc"),
                                        tf.keras.metrics.CategoricalAccuracy(name="acc")])


        hi = model.fit(x=[transposed_adjacency_train, adjacency_train, features_train, sequence_train],
                       y=[y_train, adjacency_train], validation_data=([transposed_adjacency_test, adjacency_test, features_test, sequence_test], [y_test, adjacency_test]),
                       epochs=parameters['epochs'], batch_size=batch)

        h = hi.history
        print("haa", h)
        # h = {'loss': h['loss'], 'val_loss': h['val_loss']}
        #print("summary: ", model.summary())

        y_predict_location, y_predict_graph = model.predict([transposed_adjacency_test, adjacency_test, features_test, sequence_test],
                                           batch_size=batch)

        print("saida:", type(y_predict_location), len(y_predict_location), y_predict_location)

        scores = model.evaluate([transposed_adjacency_test, adjacency_test, features_test, sequence_test],
                                [y_test, adjacency_test], batch_size=batch)
        print("scores: ", scores)

        # To transform one_hot_encoding to list of integers, representing the locations
        # print("------------- Location ------------")
        # print("saida: ", y_predict_location[0].shape, y_predict_location, y_predict_location.shape)
        y_predict_location = one_hot_decoding_predicted(y_predict_location)
        y_test = one_hot_decoding_predicted(y_test)
        # print("Original: ", y_test[0], " tamanho: ", len(y_test))
        # print("previu: ", y_predict_location[0], " tamanho: ", len(y_predict_location))
        report = skm.classification_report(y_test, y_predict_location, output_dict=True)
        # print(report)
        print("finaal", class_weight)
        return h, report

    def transpose_matrices(self, matrices):

        transposed_matrices = []

        for m in matrices:
            transposed_matrices.append(m.T)

        return np.array(transposed_matrices)




        