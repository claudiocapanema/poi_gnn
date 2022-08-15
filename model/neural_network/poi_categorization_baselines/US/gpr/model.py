import tensorflow as tf
from tensorflow.python.framework import tensor_shape
from tensorflow.keras.layers import Concatenate, Dense, Dropout, Layer, LayerNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dropout
from tensorflow.keras.regularizers import l2
from spektral.layers.convolutional import ARMAConv
from tensorflow.keras import activations, initializers, regularizers, constraints

iterations = 1          # Number of iterations to approximate each ARMA(1)
order = 2               # Order of the ARMA filter (number of parallel stacks)
share_weights = True    # Share weights in each ARMA stack
dropout = 0.5           # Dropout rate applied between layers
dropout_skip = 0.75     # Dropout rate for the internal skip connection of ARMA
l2_reg = 5e-5           # L2 regularization rate
learning_rate = 1e-2    # Learning rate
epochs = 15          # Number of training epochs
es_patience = 100       # Patience for early stopping

# @tf.function
# def divide(input):
#
#     e = input[0]
#     n = input[1]
#
#     tf.print("entradas2", e, n)
#     p = []
#
#     for i in range(e.shape[1]):
#         tensor = []
#         for j in range(e[i].shape[0]):
#             aux = e[i][j]/n[i][j]
#             tensor.append(aux)
#
#         p.append(tensor)
#
#     return p

class GGLR_unit(Layer):
    def __init__(self, main_channel):

        super(GGLR_unit, self).__init__()
        self.main_channel = main_channel

        self.main_kernel = Dense(main_channel)
        self.normalization = LayerNormalization()

    def call(self, inputs, **kwargs):

        # equação 1
        x = inputs[0]
        #x_sum = tf.reduce_sum(x, axis=1)
        x_sum = tf.keras.backend.sum(x, axis=1)
        x_sum = x_sum + 0.0000000000001
        x_out = self.main_kernel(x)

        x_out = self.normalization(x_out)
        #x_out = tf.keras.layers.Lambda(divide)([x_out, x_sum])
        x_out = tf.keras.activations.relu(x_out)

        return x_out

class Equation_4(Layer):
    def __init__(self, units=7):
        super(Equation_4, self).__init__()
        self.a = tf.Variable(1., trainable=True)
        self.b = tf.Variable(1., trainable=True)
        self.c = tf.Variable(1., trainable=True)
        self.w = Dense(units, use_bias=True, trainable=True)
        self.w2 = Dense(units, use_bias=True, trainable=True)

    def call(self, inputs, **kwargs):

        ingoing = inputs[0]
        outgoing = inputs[1]
        distance = inputs[2]

        #c_d = self.c*distance
        #f = c_d
        f = self.a*distance
        #f = tf.math.exp(c_d)
        #f = self.a*(tf.math.pow(distance, self.b)*tf.math.exp(c_d))
        #out = f*(self.w(outgoing))*ingoing
        out = (self.w(outgoing))
        # ajustar ingoing para o tamanho certo
        out = tf.multiply(out, self.w2(ingoing))
        out = tf.matmul(f, out)

        return out


class GPR_unit(Layer):
    def __init__(self, main_channel,
                 kernel_initializer='glorot_uniform',
                 kernel_regularizer=None,
                 bias_initializer='zeros',
                 bias_regularizer=None,
                 bias_constraint=None,
                 kernel_constraint=None):

        super(GPR_unit, self).__init__()
        self.main_channel = main_channel

        self.dense1 = Dense(main_channel, use_bias=False)
        self.dense2 = Dense(main_channel)

    def call(self, inputs, **kwargs):

        # equação 7
        p_out = inputs[0]
        user = inputs[1]

        #user = tf.matmul(user, self.main_kernel) # 10x10
        user_out = self.dense1(user)

        # p_out = tf.matmul(p_out, self.secondary_kernel) # 10x
        # p_out = tf.add(p_out, self.main_bias)
        p_out_out = self.dense2(p_out)
        user_out = tf.multiply(user_out, p_out_out)

        return tf.keras.activations.relu(user_out)

class GPR_component(Layer):
    def __init__(self, main_channel, units):

        super(GPR_component, self).__init__()
        self.main_channel = main_channel

        self.gglr_unit_in = GGLR_unit(main_channel)
        self.gglr_unit_out = GGLR_unit(main_channel)
        self.equation_4 = Equation_4(units)
        self.glr_unit = GPR_unit(main_channel)

    def call(self, inputs, **kwargs):

        adjacency_transposed_in = inputs[0]
        adjacencey_in = inputs[1]
        distance = inputs[2]
        user = inputs[3]

        p_in_out = self.gglr_unit_in([adjacency_transposed_in])
        p_out_out = self.gglr_unit_out([adjacencey_in])
        predicted_adjacency_matrix = self.equation_4([p_in_out, p_out_out, distance])
        gpr_out = self.glr_unit([p_out_out, user])

        return (p_in_out, p_out_out, gpr_out, predicted_adjacency_matrix)
        #return [tf.constant(1, shape=(10,10)), tf.constant(1, shape=(10,10)), tf.constant(1, shape=(10,10))]

class exibir(Layer):
    def __init__(self, units=1):
        super(exibir, self).__init__()


    def call(self, inputs):

        x = inputs[0]
        x1 = inputs[1]
        #tf.print("camada: ", x)

        return x1

class GPRUSModel:
    def __init__(self, classes, max_size_matrices, features_num_columns: int):
        self.max_size_matrices = max_size_matrices
        self.classes = classes
        self.features_num_columns = features_num_columns

    def build(self, units1=60, output_size=7, dropout=0.5, seed=None):
        if seed is not None:
            tf.random.set_seed(seed)
        A_input = Input((self.max_size_matrices, self.max_size_matrices))
        A_transposed_input = Input((self.max_size_matrices, self.max_size_matrices))
        D_input = Input((self.max_size_matrices, self.features_num_columns))
        W_input = Input((self.max_size_matrices, 7))
        U_input = Input((7))

        u_cat = Dense(20, activation='relu')(U_input)

        p_in_out1, p_out_out1, gpr_out1, predicted_adjacency_matrix = GPR_component(200, self.max_size_matrices)([A_transposed_input, A_input, D_input, u_cat])

        print("gfggg", predicted_adjacency_matrix.shape)

        # p_in_out1 = Dropout(0.4)(p_in_out1)
        # p_out_out1 = Dropout(0.4)(p_out_out1)
        # gpr_out1 = Dropout(0.4)(gpr_out1)

        p_in_out2, p_out_out2, gpr_out2, predicted_adjacency_matrix = GPR_component(60, self.max_size_matrices)([p_in_out1, p_out_out1, D_input, gpr_out1])

        out = Concatenate()([p_in_out2, p_out_out2])
        out_w = Dense(output_size, activation='softmax')(W_input) * tf.Variable(0.5)
        out = Dense(output_size, activation='softmax')(out)
        out = out + out_w

        print("aaa")
        print(out.shape)
        print("ddd")
        print(predicted_adjacency_matrix.shape)

        model = Model(inputs=[A_transposed_input, A_input, D_input, U_input, W_input], outputs=out)

        return model


