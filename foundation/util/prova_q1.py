import numpy as np

from metodos_quantitativos import MetodosQuantitativos, Metodo2kr, Metodo22r, UmFator
from curvilinear import Curvilinear
from regressao_linear import RegressaoLinear
from regressao_multilinear import RegressaoMultilinear
from sklearn.linear_model import LinearRegression
import statistics as st

if __name__ == "__main__":

    m = MetodosQuantitativos()

    print("\n---- Questão 1")

    confidence = 0.95
    a = [6, 5, 6, 7, 5, 6, 7, 8, 8, 7, 6, 7, 8, 7, 6]
    # print("/IC de Xulambis/")
    # m._intervalo_de_confianca_de_dois_lados(a=a, confidence=confidence)

    x_a = st.mean(a)
    variance_a = st.stdev(a)*st.stdev(a)
    n_a = len(a)
    x_b = 7
    variance_b = 1*1
    n_b = 15

    m.t_test(confidence=confidence, n_a=n_a, n_b=n_b, x_a=x_a, x_b=x_b,
             variance_a=variance_a, variance_b=variance_b, test_type="superior")


    print("\n---- Exercicio 1 d)")
    a = [6, 5, 6, 7, 5, 6, 7, 8, 8, 7]
    confidence = 0.95
    m._intervalo_de_confianca_de_dois_lados(a=a, confidence=confidence)
