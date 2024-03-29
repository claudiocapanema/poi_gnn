import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dropout
from tensorflow.keras.regularizers import l2
from spektral.layers.convolutional import ARMAConv

iterations = 1          # Number of iterations to approximate each ARMA(1)
order = 2               # Order of the ARMA filter (number of parallel stacks)
share_weights = True    # Share weights in each ARMA stack
dropout = 0.5           # Dropout rate applied between layers
dropout_skip = 0.75     # Dropout rate for the internal skip connection of ARMA
l2_reg = 5e-5           # L2 regularization rate
learning_rate = 1e-2    # Learning rate
epochs = 15          # Number of training epochs
es_patience = 100       # Patience for early stopping


class ARMAModel:
    def __init__(self, classes, max_size_matrices, features_num_columns: int):
        self.max_size_matrices = max_size_matrices
        self.classes = classes
        self.features_num_columns = features_num_columns

    def build(self, units1=40, output_size=9, dropout=0):
        A_input = Input((self.max_size_matrices, self.max_size_matrices))
        X_input = Input((self.max_size_matrices, self.features_num_columns))
        input_dim = self.features_num_columns

        gc_1 = ARMAConv(units1,
                        iterations=iterations,
                        order=order,
                        share_weights=share_weights,
                        dropout_rate=dropout_skip,
                        activation='elu',
                        gcn_activation='elu',
                        kernel_regularizer=l2(l2_reg))([X_input, A_input])
        gc_2 = Dropout(dropout)(gc_1)
        gc_2 = ARMAConv(output_size,
                        iterations=1,
                        order=1,
                        share_weights=share_weights,
                        dropout_rate=dropout_skip,
                        activation='softmax',
                        gcn_activation=None,
                        kernel_regularizer=l2(l2_reg))([gc_2, A_input])

        model = Model(inputs=[A_input, X_input], outputs=[gc_2])

        return model


