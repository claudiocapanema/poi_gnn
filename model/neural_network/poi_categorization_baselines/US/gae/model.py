from .layers import GraphConvolution, GraphConvolutionSparse, InnerProductDecoder, GraphConvolution2
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dropout


class GCNModelAEUS:
    def __init__(self, classes, max_size_matrices, features_num_columns: int):
        self.max_size_matrices = max_size_matrices
        self.classes = classes
        self.features_num_columns = features_num_columns

    def build(self, units1=40, output_size=9, dropout=0.5, seed=None):
        if seed is not None:
            tf.random.set_seed(seed)
        A_input = Input((self.max_size_matrices, self.max_size_matrices))
        X_input = Input((self.max_size_matrices, self.features_num_columns))
        input_dim = self.features_num_columns
        hidden1 = GraphConvolution(input_dim=input_dim,
                                              output_dim=units1,
                                              dropout=dropout)([A_input, X_input])

        #hidden1 = Dropout(0.5)(hidden1)
        embeddings = GraphConvolution2(input_dim=units1,
                                       output_dim=output_size,
                                       act=True,
                                       dropout=dropout)([A_input, hidden1])

        #embeddings = Dropout(0.5)(embeddings)
        z_mean = embeddings
        reconstructions = InnerProductDecoder(input_dim=output_size,
                                              act=lambda x: x)(embeddings)

        model = Model(inputs=[A_input, X_input], outputs=[reconstructions])

        return model


class GCNModelVAE(Model):
    def __init__(self, placeholders, num_features, num_nodes, features_nonzero, **kwargs):
        super(GCNModelVAE, self).__init__(**kwargs)

        self.inputs = placeholders['features']
        self.input_dim = num_features
        self.features_nonzero = features_nonzero
        self.n_samples = num_nodes
        self.adj = placeholders['adj']
        self.dropout = placeholders['dropout']
        self.build()

    def _build(self):
        self.hidden1 = GraphConvolutionSparse(input_dim=self.input_dim,
                                              output_dim=FLAGS.hidden1,
                                              adj=self.adj,
                                              features_nonzero=self.features_nonzero,
                                              act=tf.nn.relu,
                                              dropout=self.dropout,
                                              logging=self.logging)(self.inputs)

        self.z_mean = GraphConvolution(input_dim=FLAGS.hidden1,
                                       output_dim=FLAGS.hidden2,
                                       adj=self.adj,
                                       act=lambda x: x,
                                       dropout=self.dropout,
                                       logging=self.logging)(self.hidden1)

        self.z_log_std = GraphConvolution(input_dim=FLAGS.hidden1,
                                          output_dim=FLAGS.hidden2,
                                          adj=self.adj,
                                          act=lambda x: x,
                                          dropout=self.dropout,
                                          logging=self.logging)(self.hidden1)

        self.z = self.z_mean + tf.random.normal(shape=(self.n_samples, units2)) * tf.exp(self.z_log_std)

        self.reconstructions = InnerProductDecoder(input_dim=units2,
                                      act=lambda x: x,
                                      logging=self.logging)(self.z)