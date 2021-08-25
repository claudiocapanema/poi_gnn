import numpy as np

from metodos_quantitativos import MetodosQuantitativos, Metodo2kr, Metodo22r, UmFator
from curvilinear import Curvilinear
from regressao_linear import RegressaoLinear
from regressao_multilinear import RegressaoMultilinear
from sklearn.linear_model import LinearRegression

if __name__ == "__main__":

    m = MetodosQuantitativos()

    print("\n Questão 2")
    confidence = 0.9
    r_results = np.array([[5,4,6],
                          [7,8,9],
                          [6,7,6],
                          [10,9,9]])
    r = len(r_results[0])
    m = Metodo22r(confidence, r, r_results)
    m.contra(c_q0=0, c_qa=1, c_qb=0, c_qab=-1, confidence=confidence)