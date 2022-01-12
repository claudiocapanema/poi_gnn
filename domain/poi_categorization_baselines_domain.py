import numpy as np
import pandas as pd
import spektral as sk
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
import sklearn.metrics as skm
from tensorflow.keras import utils as np_utils

from utils.nn_preprocessing import one_hot_decoding_predicted
from domain.poi_categorization_domain import PoiCategorizationDomain
from model.neural_network.poi_categorization_baselines.BR.gae.model import GCNModelAE
from model.neural_network.poi_categorization_baselines.BR.arma.model import ARMAModel
from model.neural_network.poi_categorization_baselines.BR.arma_enhanced.model import ARMAEnhancedModel
from model.neural_network.poi_categorization_baselines.BR.gcn.model import GCN
from model.neural_network.poi_categorization_baselines.BR.gat.model import GAT
from model.neural_network.poi_categorization_baselines.BR.diffconv.model import DiffConv

from model.neural_network.poi_categorization_baselines.US.gae.model import GCNModelAEUS
from model.neural_network.poi_categorization_baselines.US.arma.arma_us import ARMAUSModel
from model.neural_network.poi_categorization_baselines.US.arma_enhanced.model import ARMAUSEnhancedModel
from model.neural_network.poi_categorization_baselines.US.gcn.model import GCNUS
from model.neural_network.poi_categorization_baselines.US.gat.model import GATUS
from model.neural_network.poi_categorization_baselines.US.diffconv.model import DiffConvUS


class PoiCategorizationBaselinesDomain(PoiCategorizationDomain):


    def __init__(self, dataset_name):
        super().__init__(dataset_name)

    def find_model(self, country, model, num_classes, max_size, features_num_columns, class_weight):
        if country == 'BR' or country == 'Brazil':
            if model == "gae":
                return GCNModelAE(num_classes, max_size, features_num_columns)
            elif model == "arma":
                return ARMAModel(num_classes, max_size, features_num_columns)
            elif model == "arma_enhanced":
                return ARMAEnhancedModel(num_classes, max_size, features_num_columns)
            elif model == "gcn":
                return GCN(num_classes, max_size, features_num_columns)
            elif model == "gat":
                return GAT(num_classes, max_size, features_num_columns)
            elif model == "diff":
                return DiffConv(num_classes, max_size, features_num_columns)
        elif country == 'US':
            if model == "gae":
                return GCNModelAEUS(num_classes, max_size, features_num_columns)
            elif model == "arma":
                return ARMAUSModel(num_classes, max_size, features_num_columns, class_weight)
            elif model == "arma_enhanced":
                return ARMAUSEnhancedModel(num_classes, max_size, features_num_columns)
            elif model == "gcn":
                return GCNUS(num_classes, max_size, features_num_columns)
            elif model == "gat":
                return GATUS(num_classes, max_size, features_num_columns)
            elif model == "diff":
                return DiffConvUS(num_classes, max_size, features_num_columns)

    def preprocess_adjacency_matrix_by_gnn_type(self, matrices, model_name):

        new_matrices = []
        if model_name == "gcn" or model_name == "gae":
            for i in range(len(matrices)):
                new_matrices.append(sk.layers.GCNConv.preprocess(matrices[i]))
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
                                                                    base_report,
                                                                    parameters,
                                                                    model_name,
                                                                    units,
                                                                    country):

        folds_histories = []
        folds_reports = []
        iteration = 0
        for i in range(len(folds)):

            fold = folds[i]
            class_weight = classes_weights[i]
            histories = []
            reports = []
            for j in range(n_replications):

                history, report = self.train_and_evaluate_baseline_model(fold,
                                                        max_size_matrices,
                                                        parameters,
                                                        model_name,
                                                        class_weight,
                                                        units,
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
                                          model_name,
                                          class_weight,
                                          units,
                                          country,
                                          seed=None):

        adjacency_train, y_train, temporal_train, \
        adjacency_test, y_test, temporal_test = fold



        print("Tamanho dados de treino: ", adjacency_train.shape, temporal_train.shape,
              y_train.shape)
        print("adjancecy: ", type(adjacency_train[0]), adjacency_train[0].shape)
        for i in range(len(adjacency_train)):
            if adjacency_train[i].shape != adjacency_train[0].shape:
                print("paraaa")
                exit()
        print("Tamanho dados de teste: ", adjacency_test.shape, temporal_test.shape,
              y_test.shape)

        num_classes = max(y_train.flatten()) + 1
        max_size = max_size_matrices
        print("Quantidade de classes: ", num_classes)
        if country == 'BR' or country == 'Brazil':
            batch = max_size * 5
        elif country == 'US':
            batch = max_size * 1

        #batch = 40
        print("Tamanho do batch: ", batch)
        print("Épocas: ", parameters['epochs'])
        print("Modelo: ", model_name)
        model = self.find_model(country, model_name, num_classes, max_size, self.features_num_columns, class_weight).build(units1=units, output_size=num_classes, seed=seed)
        y_train = np_utils.to_categorical(y_train, num_classes=num_classes)
        y_test = np_utils.to_categorical(y_test, num_classes=num_classes)
        model.compile(optimizer=parameters['optimizer'],
                      loss=parameters['loss'],
                      weighted_metrics=[tf.keras.metrics.CategoricalAccuracy(name="acc")])

        print("peso: ", class_weight)
        hi = model.fit(x=[adjacency_train, temporal_train],
                       y=y_train, validation_data=([adjacency_test, temporal_test], y_test),
                       epochs=parameters['epochs'], batch_size=batch,
                       shuffle=False,  # Shuffling data means shuffling the whole graph
                       callbacks=[
                           EarlyStopping(patience=100, restore_best_weights=True)
                       ]
                       )

        h = hi.history
        #print("summary: ", model.summary())

        y_predict_location = model.predict([adjacency_test, temporal_test],
                                           batch_size=batch)

        scores = model.evaluate([adjacency_test, temporal_test],
                                y_test, batch_size=batch)
        print("scores: ", scores)

        # To transform one_hot_encoding to list of integers, representing the locations
        # print("------------- Location ------------")
        y_predict_location = one_hot_decoding_predicted(y_predict_location)
        y_test = one_hot_decoding_predicted(y_test)
        report = skm.classification_report(y_test, y_predict_location, output_dict=True)
        # print(report)
        return h, report




        