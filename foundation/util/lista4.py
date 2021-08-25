import numpy as np

from metodos_quantitativos import MetodosQuantitativos, Metodo2kr, UmFator
from curvilinear import Curvilinear
from regressao_linear import RegressaoLinear
from regressao_multilinear import RegressaoMultilinear

if __name__ == "__main__":
    print(" - - Exercício 4")

    # 1
    # a
    # b
    # padrao
    confidence = 0.9
    x = np.array([[1, 1, 1, 1, 1, 1, 1, 1],
                  [5, 13, 20, 28, 41, 49, 61, 62],
                  [118, 132, 119, 153, 91, 118, 132, 105]]).T
    y = np.array([8.1, 6.8, 7, 7.4, 7.7, 7.5, 7.6, 8]).T
    r = RegressaoMultilinear(x=x, y=y, log=False, confidence=confidence)


    print(" - - Exercício 2 ")

    confidence = 0.9
    # x = np.array([3,5,7,9,10])
    # y = np.array([1.19,1.73,2.53,2.89,3.26])

    x = np.array([8.5, 8.9, 10.6, 11.6, 13, 13.2])
    y = np.array([30.9, 32.7, 36.7, 46.3, 46.2, 47.8])

    RegressaoLinear(x=x, y=y, confidence=confidence)

    print(" - - Exercício 3 ")

    # padrao
    confidence = 0.9
    # x = np.array([3,5,7,9,10])
    # y = np.array([1.19,1.73,2.53,2.89,3.26])
    # r = RegressaoLinear(x=x, y=y, log=False, confidence=confidence)
    # r.previsao(np.array([8]), confidence)


    x = np.array([128, 256, 512, 1024])
    y = np.array([93, 478, 3408, 25410])
    r = RegressaoLinear(x=x, y=y, log=True, confidence=confidence)
    r.previsao(np.array([384]), confidence)

    print(" - - Exercício 4")

    # 1
    # a
    # b
    # padrao
    x = np.array([[1,1,1,1,1,1,1,1],
                  [5,13,20,28,41,49,61,62],
                  [118,132,119,153,91,118,132,105]]).T
    y = np.array([8.1,6.8,7,7.4,7.7,7.5,7.6,8]).T
    r = RegressaoMultilinear(x=x, y=y, log=False, confidence=confidence)


    # b = np.array([-0.1614, 0.1182, 0.0165])
    # c = np.array([0.6297, 0.0280, 0.0012])
    # se = 1.2
    # n = 7
    # r = RegressaoMultilinear(b=b, c=c, se=se, n=n, log=False, confidence=confidence)


