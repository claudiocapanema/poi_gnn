import numpy as np

from metodos_quantitativos import MetodosQuantitativos, Metodo2kr, Metodo22r, UmFator
from curvilinear import Curvilinear
from regressao_linear import RegressaoLinear
from regressao_multilinear import RegressaoMultilinear
from sklearn.linear_model import LinearRegression

if __name__ == "__main__":

    m = MetodosQuantitativos()

    print("\n---- Exercício 2")

    confidence = 0.95
    x = np.array([8.5, 8.9,10.6,11.6,13,13.2])
    y = np.array([30.9, 32.7, 36.7, 46.3, 46.2, 47.8])
    RegressaoLinear(confidence=confidence, x=x, y=y)

    print("\n---- Exercício 3")
    confidence = 0.9
    x = np.array([128, 256, 512, 1024])
    y = np.array([93,478,3408,25410])
    r = RegressaoLinear(confidence=confidence, x=x, y=y, logy=True)
    confidence = 0.4
    r.previsao(x=np.array([384]), confidence=confidence)