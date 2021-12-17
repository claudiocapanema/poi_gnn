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
dropout_skip = 0.4    # Dropout rate for the internal skip connection of ARMA
l2_reg = 5e-5           # L2 regularization rate
learning_rate = 1e-2    # Learning rate
epochs = 15          # Number of training epochs
es_patience = 100       # Patience for early stopping

class AdaptativeGCN(Layer):
    def __init__(self, main_channel,
                 secondary_channel,
                 kernel_initializer='glorot_uniform',
                 kernel_regularizer=None,
                 bias_initializer='zeros',
                 bias_regularizer=None,
                 bias_constraint=None,
                 kernel_constraint=None):

        super(AdaptativeGCN, self).__init__(dynamic=True)
        self.main_channel = main_channel
        self.secondary_channel = secondary_channel

        self.main_layer = ARMAConv(self.main_channel,
                                iterations=1,
                                order=1,
                                share_weights=share_weights,
                                dropout_rate=dropout_skip,
                                activation=None,
                                gcn_activation=None,
                                kernel_regularizer=l2(l2_reg))
        self.main2_layer = ARMAConv(self.main_channel,
                                        iterations=1,
                                        order=1,
                                        share_weights=share_weights,
                                        dropout_rate=dropout_skip,
                                        activation=None,
                                        gcn_activation=None,
                                        kernel_regularizer=l2(l2_reg))
        self.secondary_layer = ARMAConv(self.secondary_channel,
                        iterations=iterations,
                        order=order,
                        share_weights=True,
                        dropout_rate=dropout_skip,
                        activation='relu',
                        gcn_activation='relu',
                        kernel_regularizer=l2(l2_reg))

    def get_config(self):
        config = {
            # 'bias_regularizer': self.bias_regularizer,
            # 'bias_constraint': self.bias_constraint,
                'main_layer': self.main_layer,
                'secondary_layer': self.secondary_layer,
                'main2_layer': self.main2_layer
                # 'main_use_bias': self.main_use_bias,
                # 'secondary_use_bias': self.secondary_use_bias,
                # 'main_activation': self.main_activation,
                # 'secondary_activation': self.secondary_activation,
                # 'bias_initializer': self.bias_initializer,
                # 'kernel_initializer': self.kernel_initializer,
                # 'kernel_regularizer': self.kernel_regularizer,
                # 'kernel_constraint': self.kernel_constraint,
                }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def build(self, input_shape):
        assert len(input_shape) >= 2
        input_dim = input_shape[0][-1]


        self.v_bias_out = tf.Variable(1., trainable=True)
        self.v_bias_out2 = tf.Variable(0., trainable=True)

    def compute_output_shape(self, input_shape):
        features_shape = input_shape[0]
        output_shape = features_shape[:-1] + (self.main_channel,)
        return output_shape

    def main_convolution(self, features, fltr):

        # Convolution
        output = ops.dot(features, self.main_kernel)
        output = ops.filter_dot(fltr, output)

        if self.main_use_bias:
            output = K.bias_add(output, self.main_bias)
        if self.activation is not None:
            output = self.main_activation(output)
        return output

    def secondary_convolution(self, features, fltr):

        # Convolution
        output = ops.dot(features, self.secondary_kernel)
        output = ops.filter_dot(fltr, output)

        if self.main_use_bias:
            output = K.bias_add(output, self.secondary_bias)
        if self.activation is not None:
            output = self.secondary_activation(output)
        return output

    def call(self, inputs):

        temporal_features = inputs[0]
        A = inputs[1]
        path_features = inputs[2]

        secondary = self.secondary_layer([temporal_features, A])
        secondary = tf.keras.layers.concatenate([secondary, path_features])
        main_secondary = self.main2_layer([secondary, A])

        main = tf.keras.layers.concatenate([temporal_features, path_features])

        main = self.main_layer([main, A])

        first = (self.v_bias_out2) * main
        second = (self.v_bias_out) * main_secondary

        # first = (average_lambda + tf.keras.activations.sigmoid(self.v_bias_out2)) * main
        # #tf.print("primeiro", (average_lambda + tf.keras.activations.sigmoid(self.v_bias_out2)))
        # second = ((1 - average_lambda) + tf.keras.activations.sigmoid(self.v_bias_out)) * main_secondary
        # #tf.print("segundo", ((1 - average_lambda) + tf.keras.activations.sigmoid(self.v_bias_out)))

        output = K.stack([first, second], axis=-1)
        output = K.mean(output, axis=-1)

        output = tf.keras.activations.softmax(output)

        return output


class GNNUS:

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
        # kernel_channels = 2

        out_temporal = ARMAConv(20, activation='relu')([Temporal_input, A_input])
        out_temporal = Dropout(0.5)(out_temporal)
        out_temporal = ARMAConv(self.classes,
                                activation="softmax")([out_temporal, A_input])

        out_week_temporal = ARMAConv(20, activation='relu')([Temporal_week_input, A_week_input])
        out_week_temporal = Dropout(0.5)(out_week_temporal)
        out_week_temporal = ARMAConv(self.classes,
                                activation="softmax")([out_week_temporal, A_week_input])

        out_weekend_temporal = ARMAConv(20, activation='relu')([Temporal_weekend_input, A_weekend_input])
        out_weekend_temporal = Dropout(0.5)(out_weekend_temporal)
        out_weekend_temporal = ARMAConv(self.classes,
                                     activation="softmax")([out_weekend_temporal, A_weekend_input])

        out_distance = ARMAConv(20, activation='relu')([Distance_input, A_input])
        out_distance = Dropout(0.5)(out_distance)
        out_distance = ARMAConv(self.classes,
                                activation="softmax")([out_distance, A_input])

        # out_week_distance = ARMAConv(20, activation='relu')([Distance_week_input, A_week_input])
        # out_week_distance = Dropout(0.5)(out_week_distance)
        # out_week_distance = ARMAConv(self.classes,
        #                         activation="softmax")([out_week_distance, A_week_input])
        #
        # out_weekend_distance = ARMAConv(20, activation='relu')([Distance_weekend_input, A_weekend_input])
        # out_weekend_distance = Dropout(0.5)(out_weekend_distance)
        # out_weekend_distance = ARMAConv(self.classes,
        #                         activation="softmax")([out_weekend_distance, A_weekend_input])

        out_duration = ARMAConv(20, activation='relu')([Duration_input, A_input])
        out_duration = Dropout(0.5)(out_duration)
        out_duration = ARMAConv(self.classes,
                                activation="softmax")([out_duration, A_input])

        # out_week_duration = ARMAConv(20, activation='relu')([Duration_week_input, A_week_input])
        # out_week_duration = Dropout(0.5)(out_week_duration)
        # out_week_duration = ARMAConv(self.classes,
        #                         activation="softmax")([out_week_duration, A_week_input])
        #
        # out_weekend_duration = ARMAConv(20, activation='relu')([Duration_weekend_input, A_weekend_input])
        # out_weekend_duration = Dropout(0.5)(out_weekend_duration)
        # out_weekend_duration = ARMAConv(self.classes,
        #                         activation="softmax")([out_weekend_duration, A_weekend_input])

        out = tf.Variable(1.) * out_temporal + tf.Variable(1.) * out_week_temporal + tf.Variable(1.) * out_weekend_temporal + tf.Variable(1.) * out_distance + tf.Variable(1.) * out_duration

        model = Model(inputs=[A_input, A_week_input, A_weekend_input, Temporal_input, Temporal_week_input, Temporal_weekend_input, Distance_input, Distance_week_input, Distance_weekend_input, Duration_input, Duration_week_input, Duration_weekend_input], outputs=[out])

        return model

