import math
import scipy
import statistics as st

import numpy as np
import pandas as pd
from scipy.stats import f_oneway
import seaborn as sns
from t_distribution import T_Distribution
import matplotlib.pyplot as plt

class RegressaoMultilinear:

    def __init__(self, confidence, x=None, y=None, c=None, b=None, n=None, se=None, log=False, decimals=6):
        # coluna é a variável
        self.log = log
        self.decimals = decimals
        self.confidence = confidence
        self.r2 = float
        if log:
            x = np.array([math.log(i, math.e) for i in x])
            y = np.array([math.log(i, math.e) for i in y])
        self.t_distribution = T_Distribution()

        if c is not None and b is not None and n is not None and se is not None:
            self.c = c
            self.se = se
            self.b = b
            self.b0 = b[0]
            self.b1 = b[1]
            self.b2 = b[2]
            self.n = n
            self.parametros = len(b)
            self.se = se
        else:
            self.x = x
            self.y = y
            self.n = len(x)
            self.parametros = self.x.shape[1]
            self.b = None
            self.c = None
            self.se = float
            self.b0 = float
            self.b1 = float
            self.b2 = float
            self.ssy = np.round(np.sum(np.power(self.y, 2)), self.decimals)
            self.media_y = np.round(np.mean(self.y), self.decimals)
            self.ss0 = np.round(self.n * np.power(self.media_y, 2), self.decimals)
            self.sst = np.round(self.ssy - self.ss0, self.decimals)

        self.graus_de_liberdade = self.n - self.parametros
        self.msr = float
        self.mse = float

        self.run()

    def run(self):

        if self.c is None:
            self.c = np.round(np.linalg.inv(np.matmul(self.x.T, self.x)), self.decimals)
            self.c = np.array([self.c[0,0], self.c[2,2], self.c[2,2]])
            xtx = np.matmul(self.x.T, self.x)
            bp1 = np.linalg.inv(np.matmul(self.x.T, self.x))
            bp2 = np.matmul(self.x.T, self.y)
            #print("bp1: ", bp1, " \nbp2: ", bp2)
            self.b = np.round(np.matmul(bp1, bp2), self.decimals)



        self.sse = np.round(np.matmul(self.y.T, self.y) - np.matmul(np.matmul(self.b.T, self.x.T), self.y), self.decimals)
        self.ssr = np.round(self.sst - self.sse, self.decimals)
        self.r2 = np.round(self.ssr/self.sst, self.decimals)

        self.se = np.round(np.sqrt(self.sse/(self.n - self.parametros)), self.decimals)
        self.sb0 = np.round(self.se * np.sqrt(self.c[0]), self.decimals)
        self.sb1 = np.round(self.se * np.sqrt(self.c[1]), self.decimals)
        self.sb2 = np.round(self.se * np.sqrt(self.c[2]), self.decimals)

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

        ic_min = self.b[0] - self.valor_t_ou_z * self.sb0
        ic_max = self.b[0] + self.valor_t_ou_z * self.sb0
        self.ic_b0 = np.round(np.array([ic_min, ic_max]), self.decimals)

        ic_min = self.b[1] - self.valor_t_ou_z * self.sb1
        ic_max = self.b[1] + self.valor_t_ou_z * self.sb1
        self.ic_b1 = np.round(np.array([ic_min, ic_max]), self.decimals)

        ic_min = self.b[2] - self.valor_t_ou_z * self.sb2
        ic_max = self.b[2] + self.valor_t_ou_z * self.sb2
        self.ic_b2 = np.round(np.array([ic_min, ic_max]), self.decimals)

        self.msr = self.ssr/(self.parametros - 1)
        self.mse = self.sse/(self.n - (self.parametros))
        self.fc = self.msr/self.mse

        print("X_t: ", self.x.T)
        print("X_t*X: ", xtx)
        print("C (X_t*X)^-1): ", self.c)
        print("B (X_t*X)^-1*X_t*y): ", self.b)
        print("SSE (y_t*y - b_t*X_t*y): ", self.sse)
        print("SSY (soma y^2): ", self.ssy)
        print("SS0 (n*(media y)^2): ", self.ss0)
        print("SST (SSY - SS0): ", self.sst)
        print("SSR (SST - SSE): ", self.ssr)
        print("R2 (SSR/SST): ", self.r2)
        print("Se (raiz(SSE/(n - 3)): ", self.se)
        print("C ((X_t*X)^(-1): ", self.c)
        print("Desvio Sb0 (Se*raiz(c00)): ", self.sb0)
        print("Desvio Sb1 (Se*raiz(c11)): ", self.sb1)
        print("Desvio Sb2 (Se*raiz(c22)): ", self.sb2)
        print("IC b0 (b0 +- t*sb0: ", self.ic_b0)
        print("IC b1 (b1 +- t*sb1: ", self.ic_b1)
        print("IC b2 (b2 +- t*sb2: ", self.ic_b2)

        print("MSR (SSR/k): ", self.msr)
        print("MSE (SSE/(n - (k + 1))): ", self.mse)
        print("F calculado (MSR/MSE): ", self.fc)
        print("F tabela de : ", " confianca: ", self.confidence, " n = ", self.parametros-1, " m = ", self.n - self.parametros)
