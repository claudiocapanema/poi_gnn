import math
import scipy
import statistics as st

import numpy as np
import pandas as pd
from scipy.stats import f_oneway
import seaborn as sns
from t_distribution import T_Distribution
import matplotlib.pyplot as plt

class Curvilinear:

    def __init__(self, confidence, x, y, decimals=6):
        # coluna é a variável
        self.t_distribution = T_Distribution()
        self.decimals = decimals
        self.confidence = confidence
        #self.replications = y.shape[1]
        self.x = x
        self.y = y



        self.plot()

    def plot(self):

        # x = self.x
        # y = self.y.flatten()
        # y_mean = np.mean(self.y, axis=1)
        # x_mean = x
        #
        # x_repeated = []
        # for i in x:
        #     x_repeated+=[i]*self.replications
        #
        # x_repeated = np.array(x_repeated)
        # print("x:", x_repeated, len(x_repeated))
        # print("y:", y, len(y))
        # df = pd.DataFrame({'Número de camadas': x_repeated, 'Acurácia': y})
        df = pd.DataFrame({'Acurácia': self.y, 'Número de camadas': self.x})
        figure = sns.scatterplot(x='Número de camadas', y='Acurácia', data=df)
        #plt.scatter(x=x_mean, y=y_mean)
        figure = figure.get_figure()
        figure.savefig("curvilinear2.png", bbox_inches='tight', dpi=400)