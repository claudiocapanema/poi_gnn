import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dropout
from tensorflow.keras.regularizers import l2
from spektral.layers.convolutional import ARMAConv, GATConv

iterations = 1          # Number of iterations to approximate each ARMA(1)
order = 2               # Order of the ARMA filter (number of parallel stacks)
share_weights = True    # Share weights in each ARMA stack
dropout = 0.5           # Dropout rate applied between layers
dropout_skip = 0.75     # Dropout rate for the internal skip connection of ARMA
l2_reg = 5e-5           # L2 regularization rate
learning_rate = 1e-2    # Learning rate
epochs = 15          # Number of training epochs
es_patience = 100       # Patience for early stopping
n_attn_heads = 7


class GATUS:
    def __init__(self, classes, max_size_matrices, max_size_sequence, features_num_columns: int):
        self.max_size_matrices = max_size_matrices
        self.max_size_sequence = max_size_sequence
        self.classes = classes
        self.features_num_columns = features_num_columns

    def build(self, units1=60, output_size=7, dropout=0.4, seed=None):
        if seed is not None:
            tf.random.set_seed(seed)
        A_input = Input((self.max_size_matrices, self.max_size_matrices))
        X_input = Input((self.max_size_matrices, self.features_num_columns))
        S_input = Input((self.max_size_matrices, self.max_size_sequence))

        graph_attention_1 = GATConv(units1,
                                           attn_heads=n_attn_heads,
                                           concat_heads=True,
                                           dropout_rate=dropout,
                                           activation='gelu',
                                           kernel_regularizer=l2(l2_reg),
                                           attn_kernel_regularizer=l2(l2_reg)
                                           )([X_input, A_input])
        #dropout_2 = Dropout(dropout)(graph_attention_1)
        graph_attention_2 = GATConv(output_size,
                                           attn_heads=1,
                                           concat_heads=False,
                                           dropout_rate=dropout,
                                           activation='softmax',
                                           kernel_regularizer=l2(l2_reg),
                                           attn_kernel_regularizer=l2(l2_reg)
                                           )([graph_attention_1, A_input])

        model = Model(inputs=[A_input, X_input, S_input], outputs=[graph_attention_2])

        return model


