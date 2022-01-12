import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dropout
from tensorflow.keras.regularizers import l2
from spektral.layers.convolutional import ARMAConv

iterations = 1         # Number of iterations to approximate each ARMA(1)
order = 1               # Order of the ARMA filter (number of parallel stacks)
share_weights = True    # Share weights in each ARMA stack
dropout = 0.5          # Dropout rate applied between layers
dropout_skip = 0.5  # Dropout rate for the internal skip connection of ARMA
l2_reg = 5e-5           # L2 regularization rate
learning_rate = 1e-2    # Learning rate
epochs = 15          # Number of training epochs
es_patience = 100       # Patience for early stopping
a = [2,1.1,0.7,1.1,1.1,1.1,1.1]


class exibir(Layer):
    def __init__(self, units=1):
        super(exibir, self).__init__()


    def call(self, inputs):

        x = inputs[0]
        x1 = inputs[1]
        tf.print("camada: ", x)

        return x1

class WeightInitializer(tf.keras.initializers.Initializer):

    def __init__(self, weight_list):

        self.weight_list = weight_list

    def __call__(self, shape, *args, **kwargs):
        tf.print("tamanho: ", shape)

        return 0


class ClassWeight(Model):

    def __init__(self, weight_list):
        super(ClassWeight, self).__init__()
        self.weight = tf.constant(weight_list)

    # def build(self, input_shape):
    #
    #     self.weight = self.add_weight(name='weight', shape=(input_shape[0], input_shape[1], input_shape[2]), initializer=WeightInitializer(self.weight_list), trainable=False)


    @tf.function
    def apply_weight(self, prediction):

        return self.weight[0] * prediction

    def call(self, inputs, training=None, mask=None):

        predictions = inputs

        return tf.map_fn(lambda e: self.apply_weight(e), elems=predictions)





class ARMAUSModel:
    def __init__(self, classes, max_size_matrices, features_num_columns: int, class_weight: list):
        self.max_size_matrices = max_size_matrices
        self.classes = classes
        self.features_num_columns = features_num_columns
        self.class_weight = [class_weight for i in range(max_size_matrices)]
        print("ppppppppppppp", len(self.class_weight))

    def build(self, units1=300, output_size=8, dropout=0.5, seed=None):
        if seed is not None:
            tf.random.set_seed(seed)
        A_input = Input((self.max_size_matrices, self.max_size_matrices))
        X_input = Input((self.max_size_matrices, self.features_num_columns))
        input_dim = self.features_num_columns

        #x = tf.keras.layers.Concatenate()([X_input, S_input])

        x = ARMAConv(units1,
                        iterations=iterations,
                        order=order,
                        share_weights=share_weights,
                        dropout_rate=dropout_skip,
                        activation='elu',
                        gcn_activation='gelu',
                        kernel_regularizer=l2(l2_reg))([X_input, A_input])
        #x = Dropout(0.2)(x)
        #tf.print("matrizz", x)
        gc_2 = ARMAConv(output_size,
                        iterations=1,
                        order=1,
                        share_weights=share_weights,
                        dropout_rate=dropout_skip,
                        activation='softmax',
                        gcn_activation=None,
                        kernel_regularizer=l2(l2_reg))([x, A_input])

        #gc_2 = ClassWeight(self.class_weight)([gc_2])

        #gc_2 = tf.multiply(tf.constant([a,a,a,a,a,a,a,a,a,a]), gc_2)

        model = Model(inputs=[A_input, X_input], outputs=[gc_2])

        return model


