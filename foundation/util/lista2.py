import numpy as np

from metodos_quantitativos import MetodosQuantitativos, Metodo2kr, Metodo22r, UmFator
from curvilinear import Curvilinear
from regressao_linear import RegressaoLinear
from sklearn.linear_model import LinearRegression

if __name__ == "__main__":

    m = MetodosQuantitativos()

    print("\n------ Exercício 1  a) - teste não pareado")

    confidence = 0.9
    a = [51, 52, 50]
    b = [48,  49, 47]
    print("/IC de dois lados/")
    m.t_test(confidence=confidence, a=a, b=b)
    print("/Limite inferior/")
    m.t_test(confidence=confidence, a=a, b=b, test_type='inferior')

    print("\n---- Exercício 1 - b) e c) 2kr")
    confidence = 0.9
    r_results = np.array([[41, 39, 42],
                          [51,52,50],
                         [63, 59, 64],
                          [48, 49, 47]])
    r = len(r_results[0])
    m = Metodo22r(confidence, r, r_results, log_y=True)

    print("\n---- Exercício 1 - d)")
    print("/10 execuções futuras/")
    m.estimativa(c_q0=1, c_qa=-1, c_qb=1, c_qab=-1, confidence=confidence, m=10)
    print("/1 execução futura/")
    m.estimativa(c_q0=1, c_qa=-1, c_qb=1, c_qab=-1, confidence=confidence, m=1)

    print("\n---- Execício 2 b) e c)")
    confidence = 0.9
    r_results = np.array([[1.4, 1.2, 1.3],
                          [0.6, 0.8, 0.7],
                         [1.7, 1.9, 1.8],
                          [1.2, 1, 1.1]])
    r = len(r_results[0])
    m = Metodo22r(confidence, r, r_results)

    print("\n---- Execício 3 (2kr) b) e c)")
    # valores aproximados (não é igual ao da resolução da lista)
    confidence = 0.9
    r_results = np.array([[98,100,102],
                          [245,249,256],
                          [45,54,52],
                          [300,301,299]])
    r = len(r_results[0])
    m = Metodo22r(confidence, r, r_results)
    m.contra(c_q0=0, c_qa=1, c_qb=-1, c_qab=0, confidence=confidence)

    print("\n------ Exercício 4 Um FATOR")

    confidence = 0.9
    # coluna é atributo
    # UmFator(confidence, np.array([[0.75, 0.5, 0.9, 0.65],
    #                                       [0.6, 0.8, 0.8, 0.8],
    #                                       [0.65, 0.6, 0.65, 0.7]]))

    # UmFator(confidence, np.array([[144, 120, 176, 288, 144],
    #                               [101, 144, 211, 288, 72],
    #                               [130, 180, 141, 374, 302]]).T)

    a = [0.75, 0.6, 0.65]
    b = [0.5, 0.8, 0.6]
    c = [0.9, 0.8, 0.65]
    d = [0.65, 0.8, 0.7]
    UmFator(confidence, np.array([a, b, c, d]).T)

