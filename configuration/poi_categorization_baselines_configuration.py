from tensorflow.keras.optimizers import Adam, RMSprop, Adadelta, SGD
import tensorflow as tf

from .base_poi_categorization_configuration import BasePoiCategorizationConfiguration

class PoICategorizationBaselinesConfiguration(BasePoiCategorizationConfiguration):

    OUTPUT_BASE_DIR = ("output_dir", "output/poi_categorization_baselines_job/", False, "output directory for the poi_categorization_job")

    MODEL_NAME = ("model_name", {'gae': "gae/", 'arma': "arma/", 'arma_enhanced': "arma_enhanced/",
                                 'gcn': "gcn/", 'gat': "gat/", 'diff': "diff/",
                                 'gpr': "gpr/"})

    EPOCHS = ("epochs", {'BR': {'gae': 15, 'arma': 12, 'arma_enhanced': 25, 'gcn': 15, 'gat': 15, 'diff': 15,
                         'gpr': 15},
                        'US': {'gae': 15, 'arma': 12, 'arma_enhanced': 10, 'gcn': 15, 'gat': 25, 'diff': 15,
                         'gpr': 15}})

    OPTIMIZER = ("optimizer", {'gae': Adam(), 'arma': Adam(), 'arma_enhanced': Adam(),
                               'gcn': Adam(), 'gat': Adam(), 'diff': Adam(),
                               'gpr': Adam()})

    LOSS = ("loss", {'gae': tf.keras.losses.CategoricalCrossentropy(),
                     'arma': tf.keras.losses.CategoricalCrossentropy(),
                     'arma_enhanced': tf.keras.losses.CategoricalCrossentropy(),
                     'gcn': tf.keras.losses.CategoricalCrossentropy(),
                     'gat': tf.keras.losses.CategoricalCrossentropy(),
                     'diff': tf.keras.losses.CategoricalCrossentropy(),
                     'gpr': tf.keras.losses.CategoricalCrossentropy()})

    UNITS = ("units", {'global_foursquare': {'BR': {'gat': 160, 'gcn': 200, 'arma': 180, 'arma_enhanced': 260, 'gae': 40},
                                             'Brazil': {'gat': 160, 'gcn': 200, 'arma': 30, 'arma_enhanced': 260, 'gae': 40},
                                             'US': {'gat': 160, 'gcn': 220, 'arma': 240, 'arma_enhanced': 120, 'gae': 40},
                                             'JP': {'gat': 80, 'gcn': 40, 'arma': 100, 'arma_enhanced': 40, 'gae': 40}},
                       'dense_foursquare': {'BRr': {'gat': 160, 'gcn': 200, 'arma': 180, 'arma_enhanced': 260, 'gae': 40},
                                  'Brazil': {'gat': 160, 'gcn': 200, 'arma': 20, 'arma_enhanced': 260, 'gae': 40},
                                  'US': {'gat': 160, 'gcn': 220, 'arma': 240, 'arma_enhanced': 120, 'gae': 40},
                                  'JP': {'gat': 80, 'gcn': 40, 'arma': 100, 'arma_enhanced': 40, 'gae': 40}},
                       'gowalla': {'US': {'gat': 160, 'gcn': 220, 'arma': 20, 'arma_enhanced': 120, 'gae': 40}}})

    PARAMETERS = ("parameters",
                  {'BR': {'gae': {'optimizer': OPTIMIZER[1]['gae'], 'epochs': EPOCHS[1]['BR']['gae'], 'loss': LOSS[1]['gae']},
                   'arma': {'optimizer': OPTIMIZER[1]['arma'], 'epochs': EPOCHS[1]['BR']['arma'], 'loss': LOSS[1]['arma']},
                   'arma_enhanced': {'optimizer': OPTIMIZER[1]['arma'], 'epochs': EPOCHS[1]['BR']['arma'], 'loss': LOSS[1]['arma']},
                    'gcn': {'optimizer': OPTIMIZER[1]['gcn'], 'epochs': EPOCHS[1]['BR']['gcn'], 'loss': LOSS[1]['gcn']},
                    'gat': {'optimizer': OPTIMIZER[1]['gat'], 'epochs': EPOCHS[1]['BR']['gat'], 'loss': LOSS[1]['gat']},
                   'diff': {'optimizer': OPTIMIZER[1]['diff'], 'epochs': EPOCHS[1]['BR']['diff'], 'loss': LOSS[1]['diff']},
                   'gpr': {'optimizer': OPTIMIZER[1]['gpr'], 'epochs': EPOCHS[1]['BR']['gpr'], 'loss': LOSS[1]['gpr']}},
                   'Brazil': {'gae': {'optimizer': OPTIMIZER[1]['gae'], 'epochs': EPOCHS[1]['BR']['gae'], 'loss': LOSS[1]['gae']},
                          'arma': {'optimizer': OPTIMIZER[1]['arma'], 'epochs': EPOCHS[1]['BR']['arma'], 'loss': LOSS[1]['arma']},
                          'arma_enhanced': {'optimizer': OPTIMIZER[1]['arma'], 'epochs': EPOCHS[1]['BR']['arma'],
                                            'loss': LOSS[1]['arma']},
                          'gcn': {'optimizer': OPTIMIZER[1]['gcn'], 'epochs': EPOCHS[1]['BR']['gcn'],
                                  'loss': LOSS[1]['gcn']},
                          'gat': {'optimizer': OPTIMIZER[1]['gat'], 'epochs': EPOCHS[1]['BR']['gat'],
                                  'loss': LOSS[1]['gat']},
                          'diff': {'optimizer': OPTIMIZER[1]['diff'], 'epochs': EPOCHS[1]['BR']['diff'],
                                   'loss': LOSS[1]['diff']},
                          'gpr': {'optimizer': OPTIMIZER[1]['gpr'], 'epochs': EPOCHS[1]['BR']['gpr'],
                                  'loss': LOSS[1]['gpr']}},
                  'US': {'gae': {'optimizer': OPTIMIZER[1]['gae'], 'epochs': EPOCHS[1]['US']['gae'], 'loss': LOSS[1]['gae']},
                   'arma': {'optimizer': OPTIMIZER[1]['arma'], 'epochs': EPOCHS[1]['US']['arma'], 'loss': LOSS[1]['arma']},
                   'arma_enhanced': {'optimizer': OPTIMIZER[1]['arma'], 'epochs': EPOCHS[1]['US']['arma'], 'loss': LOSS[1]['arma']},
                    'gcn': {'optimizer': OPTIMIZER[1]['gcn'], 'epochs': EPOCHS[1]['US']['gcn'], 'loss': LOSS[1]['gcn']},
                    'gat': {'optimizer': OPTIMIZER[1]['gat'], 'epochs': EPOCHS[1]['US']['gat'], 'loss': LOSS[1]['gat']},
                   'diff': {'optimizer': OPTIMIZER[1]['diff'], 'epochs': EPOCHS[1]['US']['diff'], 'loss': LOSS[1]['diff']},
                   'gpr': {'optimizer': OPTIMIZER[1]['gpr'], 'epochs': EPOCHS[1]['US']['gpr'], 'loss': LOSS[1]['gpr']}}})