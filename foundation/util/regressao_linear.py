import math
import scipy
import statistics as st

import numpy as np
import pandas as pd
from scipy.stats import f_oneway
import seaborn as sns
from t_distribution import T_Distribution
import matplotlib.pyplot as plt

class RegressaoLinear:

    def __init__(self, confidence, x, y, logy=False, logx=False, decimals=6):
        # coluna é a variáve
        if logy:
            #x = np.array([math.log(i, math.e) for i in x])
            y = np.array([math.log(i, math.e) for i in y])
        if logx:
            x = np.array([math.log(i, math.e) for i in x])
        self.parametros = 2
        self.t_distribution = T_Distribution()
        self.x = x
        self.y = y
        self.n = len(x)
        self.graus_de_liberdade = self.n - 2
        self.logy = logy
        self.logx = logx
        self.decimals = decimals
        self.confidence = confidence
        self.media_y = np.round(np.mean(self.y), decimals)
        self.media_x = np.round(np.mean(self.x), decimals)
        self.ssy = np.round(np.sum(np.power(self.y, 2)), self.decimals)
        self.ss0 = np.round(np.sum(self.n*np.power(self.media_y, 2)), decimals)
        self.sse = float
        self.sst = np.round(self.ssy - self.ss0, self.decimals)
        self.ssr = float
        self.mse = float

        self.sum_y = np.round(np.sum(self.y), self.decimals)
        self.sum_x2 = np.round(np.sum(np.power(self.x, 2)), self.decimals)
        self.sum_y2 = np.round(np.sum(np.power(self.y, 2)), self.decimals)
        self.sum_xy = float
        self.b0 = float
        self.b1 = float
        self.r2 = float
        self.sb0 = float
        self.sb1 = float

        self.ic_b0 = float
        self.ic_b1 = float
        self.se = float

        self.run()

    def run(self):

        self.sum_xy = np.round(np.sum(self.x*self.y), self.decimals)

        numerador = (self.sum_xy - self.n*self.media_x*self.media_y)
        denominador = (self.sum_x2 - self.n * np.power(self.media_x, 2))
        self.b1 = np.round(numerador/denominador, self.decimals)
        self.b0 = np.round(self.media_y - self.b1 * self.media_x, self.decimals)

        print("antes: ", self.b0, self.b1)
        if self.logy:
            # b0 = a
            #self.b0 = math.log(self.b0, math.e)
            #self.b1 = math.log(self.b1, math.e)
            pass

        self.sse = np.round(self.sum_y2 - self.b0*np.sum(self.y) - self.b1*self.sum_xy, self.decimals)

        self.ssr = np.round((self.sst - self.sse), self.decimals)
        self.r2 = np.round(self.ssr/self.sst, self.decimals)

        self.mse = np.round(self.sse/(self.n - 2), self.decimals)
        self.se = np.round(math.sqrt(self.mse), self.decimals)

        self.sb0 = np.round(self.se*math.sqrt((1/self.n)+np.power(self.media_x, 2)/(self.sum_x2 - self.n*np.power(self.media_x, 2))), self.decimals)

        self.sb1 = np.round(self.se/(math.sqrt(self.sum_x2 - self.n*np.power(self.media_x, 2))), self.decimals)

        alfa = 1 - self.confidence
        column = 1 - alfa / 2

        if self.graus_de_liberdade < 30:
            print("alfa: ", alfa, " 1 - alfa/2: ", column)
            print("graus de liberdade: ", self.graus_de_liberdade)
            self.valor_t_ou_z = self.t_distribution.find_t_distribution(column, self.graus_de_liberdade)

        else:
            if column == 0.95:
                self.valor_t_ou_z = 1.645
            elif column == 0.975:
                self.valor_t_ou_z = 1.960
            else:
                self.valor_t_ou_z = float(input("Digite z_value de "+str(column)+": "))

        print("valor t ou z: ", self.valor_t_ou_z)

        ic_min = self.b0 - self.valor_t_ou_z * self.sb0
        ic_max = self.b0 + self.valor_t_ou_z * self.sb0
        self.ic_b0 = np.round(np.array([ic_min, ic_max]), self.decimals)

        ic_min = self.b1 - self.valor_t_ou_z * self.sb1
        ic_max = self.b1 + self.valor_t_ou_z * self.sb1
        self.ic_b1 = np.round(np.array([ic_min, ic_max]), self.decimals)

        self.msr = self.ssr / (self.parametros - 1)
        self.mse = self.sse / (self.n - (self.parametros))
        self.fc = self.msr / self.mse

        print("Resultados:")
        print("valor n: ", self.n)
        print("Media x: ", self.media_x, " Media y: ", self.media_y)
        print("Soma y: ", self.sum_y)
        print("Soma x2: ", self.sum_x2)
        print("Soma y2: ", self.sum_y2)
        print("Soma xy: ", self.sum_xy)
        print("B0 (media y - b1*media x): ", self.b0)
        print("B1 ((soma xy - n*media x * media y)/(soma x2 - n*media x^2)): ", self.b1)
        print("Desvios Sb0: ", self.sb0, " Sb1: ", self.sb1)
        print("SSY: ", self.ssy)
        print("SST (soma y^2(SSY) - n*(media y)^2): ", self.sst)
        print("SS0 (n*media y^2): ", self.ss0)
        print("SSE (soma y^2 - B0*soma y - B1*soma xy): ", self.sse)
        print("SSR (SST - SSE): ", self.ssr)
        print("MSE (SSE/(n-2)): ", self.mse)
        print("Se (raiz(SSE)): ", self.se)
        print("R^2 (SSR/SST): ", self.r2)

        print("Desvio Sb0 (se * raiz((1/n + ((media x)^2)/(soma x^2 - n*(media x)^2)): ", self.sb0)
        print("Desvio Sb1 (se/(raiz((soma x^2) - n*(media x)^2): ", self.sb1)

        print("IC b0 (b0 +- t*sb0): ", self.ic_b0)
        print("IC b1 (b1 +- t*sb1): ", self.ic_b1)

        print("MSR (SSR/k): ", self.msr)
        print("F calculado (MSR/MSE): ", self.fc)
        print("F tabela de : ", " confianca: ", self.confidence, " n = ", self.parametros - 1, " m = ",
              self.n - self.parametros)

        df = pd.DataFrame({'Acurácia': self.y, 'Número de camadas': self.x})
        figure = sns.scatterplot(x='Número de camadas', y='Acurácia', data=df)
        # plt.scatter(x=x_mean, y=y_mean)
        x = self.x
        y = [self.b0 + self.b1*i for i in x]
        plt.plot(x, y)

        figure = figure.get_figure()
        figure.savefig("linear.png", bbox_inches='tight', dpi=400)

    def previsao(self, x, confidence):
        m = len(x)
        print("n: ", self.n, " m: ", m)
        y = np.round(self.b0 + self.b1*x, self.decimals)
        if self.logx:
            #x = np.array([math.log(i, math.e) for i in x])
            y = np.round(self.b0 + self.b1*np.array([math.log(i, math.e) for i in x]), self.decimals)
        print("y previsto: ", y)
        print("deno: ")
        sy = np.round(self.se * np.sqrt((1/m + 1/self.n + np.power((x - self.media_x), 2)/(self.sum_x2 - self.n*np.power(self.media_x, 2)))), self.decimals)

        alfa = 1 - confidence
        column = 1 - alfa / 2

        if self.graus_de_liberdade < 30:
            print("alfa: ", alfa, " 1 - alfa/2: ", column)
            print("graus de liberdade: ", self.graus_de_liberdade)
            valor_t_ou_z = self.t_distribution.find_t_distribution(column, self.graus_de_liberdade)

        else:
            if column == 0.95:
                valor_t_ou_z = 1.645
            elif column == 0.975:
                valor_t_ou_z = 1.960
            else:
                valor_t_ou_z = float(input("Digite z_value de " + str(column) + ": "))

        print("valor t ou z: ", self.valor_t_ou_z)

        ic_min = y - valor_t_ou_z * sy
        ic_max = y + valor_t_ou_z * sy
        ic_y = np.round([ic_min, ic_max], self.decimals)

        print("Sy: ", sy)
        print("erro: ", (valor_t_ou_z*sy)/y)
        print(("IC: ", ic_y.tolist()))
