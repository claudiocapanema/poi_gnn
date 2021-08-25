import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dropout
from tensorflow.keras.regularizers import l2
from spektral.layers.convolutional import ARMAConv, GCNConv

iterations = 1          # Number of iterations to approximate each ARMA(1)
order = 2               # Order of the ARMA filter (number of parallel stacks)
share_weights = True    # Share weights in each ARMA stack
dropout = 0.7           # Dropout rate applied between layers
dropout_skip = 0.75     # Dropout rate for the internal skip connection of ARMA
l2_reg = 5e-5           # L2 regularization rate
learning_rate = 1e-2    # Learning rate
epochs = 15          # Number of training epochs
es_patience = 100       # Patience for early stopping


class GCN:
    def __init__(self, classes, max_size_matrices, max_size_sequence, features_num_columns: int):
        self.max_size_matrices = max_size_matrices
        self.max_size_sequence = max_size_sequence
        self.classes = classes
        self.features_num_columns = features_num_columns

    def build(self, units1=60, output_size=8, dropout=0.7, seed=None):
        if seed is not None:
            tf.random.set_seed(seed)
        A_input = Input((self.max_size_matrices, self.max_size_matrices))
        X_input = Input((self.max_size_matrices, self.features_num_columns))
        S_input = Input((self.max_size_matrices, self.max_size_sequence))

        graph_conv_1 = GCNConv(units1,
                                 activation='elu',
                                 kernel_regularizer=l2(l2_reg),
                                 use_bias=True)([X_input, A_input])
        #dropout_2 = Dropout(dropout)(graph_conv_1)
        graph_conv_2 = GCNConv(output_size,
                                 activation='softmax',
                                 use_bias=True)([graph_conv_1, A_input])

        model = Model(inputs=[A_input, X_input, S_input], outputs=[graph_conv_2])

        return model


