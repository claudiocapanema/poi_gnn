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
order = 2               # Order of the ARMA filter (number of parallel stacks)
share_weights = True    # Share weights in each ARMA stack
dropout = 0.5           # Dropout rate applied between layers
dropout_skip = 0.35      # Dropout rate for the internal skip connection of ARMA
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

        # self.main_layer = GCNConv(self.main_channel, activation='softmax', name='main')
        # self.main2_layer = GCNConv(self.main_channel, activation='softmax', name='main2')
        # self.secondary_layer = GCNConv(secondary_channel, activation='relu', name='secondary')

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
                        iterations=1,
                        order=1,
                        share_weights=True,
                        dropout_rate=dropout_skip,
                        activation='elu',
                        gcn_activation='elu',
                        kernel_regularizer=l2(l2_reg))



    def get_config(self):
        config = {
            # 'bias_regularizer': self.bias_regularizer,
            # 'bias_constraint': self.bias_constraint,
                'main_layer': self.main_layer,
                'secondary_layer': self.secondary_layer,
                'main2_layer': self.main2_layer,
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
        # self.main_kernel = self.add_weight(shape=(input_dim, self.main_channel),
        #                               initializer=self.kernel_initializer,
        #                               name='main_kernel',
        #                               regularizer=self.kernel_regularizer,
        #                               constraint=self.kernel_constraint)
        # if self.main_use_bias:
        #     self.bias = self.add_weight(shape=(self.main_channel,),
        #                                 initializer=self.bias_initializer,
        #                                 name='main_bias',
        #                                 regularizer=self.bias_regularizer,
        #                                 constraint=self.bias_constraint)
        # else:
        #     self.main_bias = None
        # self.main_built = True
        #
        # self.secondary_kernel = self.add_weight(shape=(input_dim, self.secondary_channel),
        #                                    initializer=self.kernel_initializer,
        #                                    name='secondary_kernel',
        #                                    regularizer=self.kernel_regularizer,
        #                                    constraint=self.kernel_constraint)
        # if self.secondary_use_bias:
        #     self.secondary_bias = self.add_weight(shape=(self.secondary_channel,),
        #                                 initializer=self.bias_initializer,
        #                                 name='secondary_bias',
        #                                 regularizer=self.bias_regularizer,
        #                                 constraint=self.bias_constraint)
        #
        # self.average_kernel = self.add_weight(shape=(self.main_channel,),
        #                                    initializer=self.kernel_initializer,
        #                                    name='main_kernel',
        #                                    regularizer=self.kernel_regularizer,
        #                                    constraint=self.kernel_constraint)

        self.v_bias = tf.Variable(1., trainable=True)
        self.v_bias_out = tf.Variable(1., trainable=True)
        self.v_bias_out2 = tf.Variable(0., trainable=True)
        self.v_bias_in = tf.Variable(1., trainable=True)
        self.v_bias_in2 = tf.Variable(0., trainable=True)

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

        x = inputs[0]
        a = inputs[1]
        user_metrics = inputs[2]
        x2 = inputs[3]

        average_lambda = tf.reduce_mean(user_metrics)
        average_lambda = self.v_bias*tf.math.pow(2.71, -(average_lambda*self.v_bias_in + self.v_bias_in2))

        secondary = self.secondary_layer([x, a])
        secondary = tf.keras.layers.concatenate([secondary, x2])
        main_secondary = self.main2_layer([secondary, a])

        main = tf.keras.layers.concatenate([x, x2])

        main = self.main_layer([main, a])

        first = (self.v_bias_out2 + average_lambda) * main
        second = (self.v_bias_out - average_lambda) * main_secondary

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
        X_input = Input((self.max_size_matrices, self.features_num_columns))
        Metrics_input = Input((1))
        S_input = Input((self.max_size_matrices, self.max_size_sequence))
        kernel_channels = 2

        # x2 = ARMAConv(220,
        #          iterations=iterations,
        #          order=order,
        #          share_weights=share_weights,
        #          dropout_rate=dropout_skip,
        #          activation='elu',
        #          gcn_activation='elu',
        #          kernel_regularizer=l2(l2_reg))([S_input, A_input])
        # x2 = Dropout(0.5)(x2)
        #

        out = AdaptativeGCN(main_channel=self.classes, secondary_channel=200)([X_input, A_input, Metrics_input, S_input])
        # out = GCNConv(self.classes, activation='softmax')([out, A_input])
        #out = Dense(self.classes, activation='softmax')(x)

        model = Model(inputs=[A_input, X_input, Metrics_input, S_input], outputs=[out])

        return model

