import tensorflow as tf
import tensorflow.keras.backend as K
import spektral as sk
from spektral.layers.convolutional import GraphAttention, GraphConv, ARMAConv, GraphConvSkip
from spektral.layers.pooling import GlobalAttentionPool, TopKPool
from tensorflow.keras.layers import Input, Dense, Masking, Dropout, Flatten, Layer
from tensorflow.keras.models import Model
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.metrics import CategoricalAccuracy
from sklearn.decomposition import non_negative_factorization
from tensorflow.keras.regularizers import l2
import numpy as np
from spektral.layers import ops
from spektral.utils.convolution import normalized_adjacency
from tensorflow.keras import activations, initializers, regularizers, constraints

class GNNTransferLearning:

    def __init__(self, classes, max_size_matrices, base_model, features_num_columns: int):
        self.max_size_matrices = max_size_matrices
        self.classes = classes
        self.base_model = base_model
        self.features_num_columns = features_num_columns

    def build(self):
        l2_reg = 5e-4 / 2  # L2 regularization rate
        A_input = Input((self.max_size_matrices,self.max_size_matrices))
        X_input = Input((self.max_size_matrices, self.features_num_columns))
        Metrics_input = Input((1))
        kernel_channels = 2

        #metrics = Dense(kernel_channels, activation='softmax')(Metrics_input)
        #print("metrics: ", metrics.shape)
        #metrics = Dropout(0.5)(metrics)
        # x = ARMAConv(40, activation='elu', kernel_regularizer=l2(l2_reg),
        #              share_weights=True, gcn_activation='elu',dropout_rate=0.75)([X_input, A_input])
        # out = GraphConv(40, activation='relu')([X_input, A_input])
        # out = Dropout(0.5)(out)

        # out = GraphConv(self.classes, activation='softmax')([out, A_input])
        #out = Dense(self.classes, activation='softmax')(x)

        model = Model(inputs=[A_input, X_input, Metrics_input], outputs=[out])

        return model

