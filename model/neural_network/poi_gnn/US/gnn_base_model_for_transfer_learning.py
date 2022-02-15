import tensorflow as tf
import tensorflow.keras.backend as K
import spektral as sk
from spektral.layers.convolutional import GCNConv, ARMAConv
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


iterations = 1          # Number of iterations to approximate each ARMA(1)
order = 1               # Order of the ARMA filter (number of parallel stacks)
share_weights = True    # Share weights in each ARMA stack
dropout = 0.5           # Dropout rate applied between layers
dropout_skip = 0.3    # Dropout rate for the internal skip connection of ARMA
l2_reg = 5e-5           # L2 regularization rate
learning_rate = 1e-2    # Learning rate
epochs = 15          # Number of training epochs
es_patience = 100       # Patience for early stopping

class GNNBaseModel(Model):

    def __init__(self):
        super(GNNBaseModel, self).__init__()

        self.temporal_layer_1 = ARMAConv(20, activation='elu',
                                gcn_activation='gelu', share_weights=share_weights,
                                dropout_rate=dropout_skip)
        self.temporal_layer_2 = ARMAConv(self.classes)

        self.week_temporal_layer_1 = ARMAConv(20, activation='elu',
                                     gcn_activation='gelu', share_weights=share_weights,
                                     dropout_rate=dropout_skip)
        self.week_temporal_layer_2 = ARMAConv(self.classes)

        self.weekend_temporal_layer_1 = ARMAConv(20, activation='elu',
                                        gcn_activation='gelu', share_weights=share_weights,
                                        dropout_rate=dropout_skip)
        self.weekend_temporal_layer_2 = ARMAConv(self.classes)

        self.distance_layer_1 = ARMAConv(20, activation='elu',
                                gcn_activation='gelu')
        self.distance_layer_2 = ARMAConv(self.classes)

        self.duration_layer_1 = ARMAConv(20, activation='elu',
                                gcn_activation='gelu')
        self.duration_layer_2 = ARMAConv(self.classes)

    def call(self, inputs, include_top=True, training=None, mask=None):

        A_input, A_week_input, A_weekend_input, Temporal_input, Temporal_week_input, Temporal_weekend_input, Distance_input, Distance_week_input, Distance_weekend_input, Duration_input, Duration_week_input, Duration_weekend_input = inputs

        out_temporal = self.temporal_layer_1([Temporal_input, A_input])
        out_temporal = Dropout(0.3)(out_temporal)
        out_temporal = self.temporal_layer_2([out_temporal, A_input])

        out_week_temporal = self.week_temporal_layer_1([Temporal_week_input, A_week_input])
        out_week_temporal = Dropout(0.3)(out_week_temporal)
        out_week_temporal = self.week_temporal_layer_2([out_week_temporal, A_week_input])

        out_weekend_temporal = self.weekend_temporal_layer_1([Temporal_weekend_input, A_weekend_input])
        out_weekend_temporal = Dropout(0.3)(out_weekend_temporal)
        out_weekend_temporal = self.weekend_temporal_layer_2([out_weekend_temporal, A_weekend_input])

        out_distance = self.distance_layer_1([Distance_input, A_input])
        out_distance = Dropout(0.3)(out_distance)
        out_distance = self.distance_layer_2([out_distance, A_input])

        out_duration = self.duration_layer_1([Duration_input, A_input])
        out_duration = Dropout(0.3)(out_duration)
        out_duration = self.duration_layer_2([out_duration, A_input])

        if include_top:
            out_temporal = tf.keras.activations.softmax(out_temporal)
            out_week_temporal = tf.keras.activations.softmax(out_week_temporal)
            out_weekend_temporal = tf.keras.activations.softmax(out_weekend_temporal)
            out_distance = tf.keras.activations.softmax(out_distance)
            out_duration = tf.keras.activations.softmax(out_duration)

        return out_temporal, out_week_temporal, out_weekend_temporal, out_distance, out_duration



class GNNUS_BaseModel:

    def __init__(self, classes, max_size_matrices, max_size_sequence, features_num_columns: int):
        self.max_size_matrices = max_size_matrices
        self.max_size_sequence = max_size_sequence
        self.classes = classes
        self.features_num_columns = features_num_columns

    def build(self, seed=None):
        if seed is not None:
            tf.random.set_seed(seed)

        l2_reg = 5e-4 / 2  # L2 regularization rate
        A_input = Input((self.max_size_matrices,self.max_size_matrices))
        A_week_input = Input((self.max_size_matrices, self.max_size_matrices))
        A_weekend_input =  Input((self.max_size_matrices, self.max_size_matrices))
        Temporal_input = Input((self.max_size_matrices, self.features_num_columns))
        Temporal_week_input = Input((self.max_size_matrices, 24))
        Temporal_weekend_input = Input((self.max_size_matrices, 24))
        Path_input = Input((self.max_size_matrices, self.max_size_sequence))
        Distance_input = Input((self.max_size_matrices,self.max_size_matrices))
        Distance_week_input = Input((self.max_size_matrices, self.max_size_matrices))
        Distance_weekend_input = Input((self.max_size_matrices, self.max_size_matrices))
        Duration_input = Input((self.max_size_matrices,self.max_size_matrices))
        A_week_input = Input((self.max_size_matrices, self.max_size_matrices))
        Duration_week_input = Input((self.max_size_matrices, self.max_size_matrices))
        Duration_weekend_input = Input((self.max_size_matrices, self.max_size_matrices))
        Location_time_input = Input((self.max_size_matrices, self.features_num_columns))
        # kernel_channels = 2

        out_temporal = ARMAConv(20, activation='elu',
                                gcn_activation='gelu', share_weights=share_weights,
                                dropout_rate=dropout_skip)([Temporal_input, A_input])
        out_temporal = Dropout(0.3)(out_temporal)
        out_temporal = ARMAConv(self.classes,
                                activation="softmax")([out_temporal, A_input])

        out_week_temporal = ARMAConv(20, activation='elu',
                                     gcn_activation='gelu', share_weights=share_weights,
                                     dropout_rate=dropout_skip)([Temporal_week_input, A_week_input])
        out_week_temporal = Dropout(0.3)(out_week_temporal)
        out_week_temporal = ARMAConv(self.classes,
                                     activation="softmax")([out_week_temporal, A_week_input])

        out_weekend_temporal = ARMAConv(20, activation='elu',
                                        gcn_activation='gelu', share_weights=share_weights,
                                        dropout_rate=dropout_skip)([Temporal_weekend_input, A_weekend_input])
        out_weekend_temporal = Dropout(0.3)(out_weekend_temporal)
        out_weekend_temporal = ARMAConv(self.classes,
                                        activation="softmax")([out_weekend_temporal, A_weekend_input])

        out_distance = ARMAConv(20, activation='elu',
                                gcn_activation='gelu')([Distance_input, A_input])
        out_distance = Dropout(0.3)(out_distance)
        out_distance = ARMAConv(self.classes,
                                activation="softmax")([out_distance, A_input])

        out_duration = ARMAConv(20, activation='elu',
                                gcn_activation='gelu')([Duration_input, A_input])
        out_duration = Dropout(0.3)(out_duration)
        out_duration = ARMAConv(self.classes,
                                activation="softmax")([out_duration, A_input])

        out_location_time = Dense(self.classes, activation='softmax')(Location_time_input)

        out = tf.Variable(1.) * out_temporal + tf.Variable(1.) * out_week_temporal + tf.Variable(1.) * out_weekend_temporal + tf.Variable(1.) * out_distance + tf.Variable(1.) * out_duration + tf.Variable(1.) * out_location_time

        model = Model(inputs=[A_input, A_week_input, A_weekend_input, Temporal_input, Temporal_week_input, Temporal_weekend_input, Distance_input, Duration_input, Location_time_input], outputs=[out])

        return model

