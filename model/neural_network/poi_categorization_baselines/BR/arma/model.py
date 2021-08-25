import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dropout
from tensorflow.keras.regularizers import l2
from spektral.layers.convolutional import ARMAConv

iterations = 1         # Number of iterations to approximate each ARMA(1)
order = 2               # Order of the ARMA filter (number of parallel stacks)
share_weights = True    # Share weights in each ARMA stack
dropout = 0.5          # Dropout rate applied between layers
dropout_skip = 0.4  # Dropout rate for the internal skip connection of ARMA
l2_reg = 5e-5           # L2 regularization rate
learning_rate = 1e-2    # Learning rate
epochs = 15          # Number of training epochs
es_patience = 100       # Patience for early stopping


class exibir(Layer):
    def __init__(self, units=1):
        super(exibir, self).__init__()


    def call(self, inputs):

        x = inputs[0]
        x1 = inputs[1]
        tf.print("camada: ", x)

        return x1



class ARMAModel:
    def __init__(self, classes, max_size_matrices, features_num_columns: int):
        self.max_size_matrices = max_size_matrices
        self.classes = classes
        self.features_num_columns = features_num_columns

    def build(self, units1=30, output_size=8, dropout=0.5, seed=None):
        print("unidades: ", units1, " saida: ", self.classes, " tamanho maximo: ", self.max_size_matrices)
        if seed is not None:
            tf.random.set_seed(seed)
        A_input = Input((self.max_size_matrices, self.max_size_matrices))
        Temporal_input = Input((self.max_size_matrices, self.features_num_columns))

        out_temporal = ARMAConv(units1, activation='elu')([Temporal_input, A_input])
        out_temporal = Dropout(0.5)(out_temporal)
        out_temporal = ARMAConv(self.classes, activation="softmax")([out_temporal, A_input])

        model = Model(inputs=[A_input, Temporal_input], outputs=[out_temporal])

        return model


