import numpy as np

from metodos_quantitativos import MetodosQuantitativos, Metodo2kr, Metodo22r, UmFator
from curvilinear import Curvilinear
from regressao_linear import RegressaoLinear
from regressao_multilinear import RegressaoMultilinear
from sklearn.linear_model import LinearRegression

if __name__ == "__main__":

    m = MetodosQuantitativos()

    print("\n---- Questão 3")

    confidence = 0.9
    a = [4,5,3]
    b = [6,7,6]
    c = [5,6,6]
    UmFator(confidence, np.array([a, b, c]).T)