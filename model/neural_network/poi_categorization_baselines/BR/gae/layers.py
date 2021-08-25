from .initializations import *
import tensorflow as tf
from tensorflow.keras.layers import Layer, Dropout
from tensorflow.keras import activations, initializers, regularizers, constraints

# global unique layer ID dictionary for layer name assignment
_LAYER_UIDS = {}

def dropout_sparse(x, keep_prob, num_nonzero_elems):
    """Dropout for sparse tensors. Currently fails for very large sparse tensors (>1M elements)
    """
    noise_shape = [num_nonzero_elems]
    random_tensor = keep_prob
    random_tensor += tf.random.uniform(noise_shape)
    dropout_mask = tf.cast(tf.floor(random_tensor), dtype=tf.bool)
    pre_out = tf.sparse_retain(x, dropout_mask)
    return pre_out * (1./keep_prob)

class GraphConvolution(Layer):
    """Basic graph convolution layer for undirected graph without edge labels."""
    def __init__(self, input_dim, output_dim, act=False, dropout=0.5,
                 kernel_initializer='glorot_uniform',
                 kernel_regularizer=None,
                 kernel_constraint=None):
        super(GraphConvolution, self).__init__()
        #self.shared_weights = weight_variable_glorot(input_dim, output_dim, name="weights")
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.act = act

        self.kernel_initializer = initializers.get(kernel_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)

    def build(self, input_shape):
        self.shared_weights = self.add_weight(shape=(self.input_dim, self.output_dim),
                        initializer=self.kernel_initializer,
                        name='main_kernel',
                        regularizer=self.kernel_regularizer,
                        constraint=self.kernel_constraint)

    def call(self, inputs):
        adj = inputs[0]
        x = inputs[1]
        tf.print("tamanho variavel: ", self.shared_weights.shape, " entrada: ", x.shape)
        feats = tf.matmul(x, self.shared_weights)
        #x = tf.sparse_tensor_dense_matmul(self.adj, x)
        tf.print("primeira: ", feats.shape)
        feats = tf.matmul(adj, feats)
        tf.print("segunda: ", feats.shape)
        outputs = activations.relu(feats)
        return outputs

class GraphConvolution2(Layer):
    """Basic graph convolution layer for undirected graph without edge labels."""
    def __init__(self, input_dim, output_dim, act=False, dropout=0.5,
                 kernel_initializer='glorot_uniform',
                 kernel_regularizer=None,
                 kernel_constraint=None):
        super(GraphConvolution2, self).__init__()
        #self.shared_weights = weight_variable_glorot(input_dim, output_dim, name="weights")
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.act = act

        self.kernel_initializer = initializers.get(kernel_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.kernel_constraint = constraints.get(kernel_constraint)

    def build(self, input_shape):
        self.shared_weights = self.add_weight(shape=(self.input_dim, self.output_dim),
                        initializer=self.kernel_initializer,
                        name='main_kernel',
                        regularizer=self.kernel_regularizer,
                        constraint=self.kernel_constraint)

    def call(self, inputs):
        adj = inputs[0]
        x = inputs[1]
        tf.print("tamanho variavel: ", self.shared_weights.shape, " entrada: ", x.shape)
        feats = tf.matmul(x, self.shared_weights)
        #x = tf.sparse_tensor_dense_matmul(self.adj, x)
        tf.print("primeira: ", feats.shape)
        feats = tf.matmul(adj, feats)
        tf.print("segunda: ", feats.shape)
        return activations.relu(feats)


class GraphConvolutionSparse(Layer):
    """Graph convolution layer for sparse inputs."""
    def __init__(self, input_dim, output_dim, adj, features_nonzero, dropout=0., act=tf.nn.relu, **kwargs):
        super(GraphConvolutionSparse, self).__init__(**kwargs)
        with tf.variable_scope(self.name + '_vars'):
            self.vars['weights'] = weight_variable_glorot(input_dim, output_dim, name="weights")
        self.dropout = dropout
        self.adj = adj
        self.act = act
        self.issparse = True
        self.features_nonzero = features_nonzero

    def _call(self, inputs):
        x = inputs
        x = dropout_sparse(x, 1-self.dropout, self.features_nonzero)
        x = tf.sparse_tensor_dense_matmul(x, self.vars['weights'])
        x = tf.sparse_tensor_dense_matmul(self.adj, x)
        outputs = self.act(x)
        return outputs


class InnerProductDecoder(Layer):
    """Decoder model layer for link prediction."""
    def __init__(self, input_dim, dropout=0., act=tf.nn.sigmoid):
        super(InnerProductDecoder, self).__init__()
        self.dropout = Dropout(1-dropout)

    def _call(self, inputs):
        #inputs = self.dropout(inputs)
        x = tf.transpose(inputs)
        x = tf.matmul(inputs, x)
        x = tf.reshape(x, [-1])
        outputs = x
        return tf.keras.activations.softmax(outputs)