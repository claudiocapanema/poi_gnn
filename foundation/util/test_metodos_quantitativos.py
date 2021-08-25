import numpy as np

from metodos_quantitativos import MetodosQuantitativos, Metodo2kr, UmFator

if __name__ == "__main__":

    m = MetodosQuantitativos()

    print("\n------ Teste 1")
    a = [5.36, 16.57, 0.62, 1.41, 0.64, 7.26]
    b = [19.12, 3.52, 3.38, 2.50, 3.60, 1.74]

    confidence = 0.9

    print(m.t_test(confidence=confidence, a=a, b=b))

    n_a = 972
    n_b = 153
    x_a = 124.1
    x_b = 141.47
    variance_a = 198.2*198.2
    variance_b = 226.11*226.11

    print("\n------ Limite superior 2")
    print(m.limite_superior(confidence=0.9,
                            n_a=n_a, n_b=n_b,
                            media_a=x_a, media_b=x_b,
                            variance_a=variance_a,
                            variance_b=variance_b,
                            z_score=1.282))

    print("\n------ Teste 1-a-1")
    a = [51, 52, 50]
    b = [48, 49, 47]

    confidence = 0.9

    print(m.observacoes_pareadas(confidence=confidence, a=a, b=b))

    print("\n------ Observações pareadas")
    a = [4,5,0,11,6,6,3,12,9,5,6,3,1,6]
    b = [2,7,7,6,0,7,10,6,2,2,4,2,2,0]

    confidence = 0.9

    print(m.observacoes_pareadas(confidence=confidence, a=a, b=b))

    print("\n------ Limite inferior 1-a-2")

    confidence = 0.95

    print(m.limite_inferior(confidence=confidence, a=a, b=b))

    print("\n------ Método 2kr 2")

    confidence = 0.9
    Metodo2kr(confidence, 2, 3, np.array([[1.4, 1.2, 1.3],
                                   [0.6, 0.8, 0.7],
                                   [1.7, 1.9, 1.8],
                                   [1.2, 1, 1.1]]))

    print("\n------ Método 2kr 3")

    confidence = 0.9
    Metodo2kr(confidence, 2, 3, np.array([[98, 100, 102],
                                          [245, 249, 256],
                                          [45, 54, 52],
                                          [300, 301, 299]]))

    print("\n------ Um fator 4")

    confidence = 0.9
    # linha é atributo
    UmFator(confidence, np.array([[0.75, 0.5, 0.9, 0.65],
                                          [0.6, 0.8, 0.8, 0.8],
                                          [0.65, 0.6, 0.65, 0.7]]))
    # UmFator(confidence, np.array([[144, 120, 176, 288, 144],
    #                               [101, 144, 211, 288, 72],
    #                               [130, 180, 141, 374, 302]]).T)
