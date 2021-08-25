import numpy as np

from metodos_quantitativos import MetodosQuantitativos, Metodo2kr, UmFator
from curvilinear import Curvilinear
from regressao_linear import RegressaoLinear
from sklearn.linear_model import LinearRegression

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

    print("\n------ Teste 2")
    print(m.limite_superior(confidence=0.9,
                            n_a=n_a, n_b=n_b,
                            media_a=x_a, media_b=x_b,
                            variance_a=variance_a,
                            variance_b=variance_b))

    print("\n------ Teste 1-a-1")
    a = [51, 52, 50]
    b = [48, 49, 47]

    confidence = 0.9

    print(m.observacoes_pareadas(confidence=confidence, a=a, b=b))

    print("\n------ Teste 1-a-2")

    confidence = 0.95

    print(m.limite_inferior(confidence=confidence, a=a, b=b))

    print("\n------ Teste 2")

    confidence = 0.9
    Metodo2kr(confidence, 2, 3, np.array([[15,18,12],
                                   [45,48,51],
                                   [25,28,19],
                                   [75,75,81]]))

    print("\n------ Teste 3")

    confidence = 0.9
    Metodo2kr(confidence, 2, 3, np.array([[98, 100, 102],
                                          [245, 249, 256],
                                          [45, 54, 52],
                                          [300, 301, 299]]))

    print("\n------ Um FATOR")

    confidence = 0.95
    # coluna é atributo
    # UmFator(confidence, np.array([[0.75, 0.5, 0.9, 0.65],
    #                                       [0.6, 0.8, 0.8, 0.8],
    #                                       [0.65, 0.6, 0.65, 0.7]]))

    # UmFator(confidence, np.array([[144, 120, 176, 288, 144],
    #                               [101, 144, 211, 288, 72],
    #                               [130, 180, 141, 374, 302]]).T)

    gcn = [0.323140625, 0.319140625, 0.318796875, 0.311171875, 0.316921875, 0.307859375, 0.31740625, 0.31603125, 0.31346875, 0.3230625]
    gat = [0.339703125, 0.338015625, 0.339203125, 0.33590625, 0.337390625, 0.3385, 0.33803125, 0.3386875, 0.339609375, 0.3405]
    arma = [0.356375, 0.350828125, 0.357640625, 0.355109375, 0.357546875, 0.355859375, 0.354109375, 0.353421875, 0.353921875, 0.355703125]
    UmFator(confidence, np.array([arma, gat, gcn]).T)

    print("\n------ 2kr")

    confidence = 0.95
    resultados = np.array([[0.35471875, 0.3468125, 0.35528125, 0.353875, 0.3536875, 0.3575625, 0.35028125, 0.35334375, 0.355125, 0.35378125],
                                          [0.350375, 0.339359375, 0.34765625, 0.346078125, 0.35059375, 0.351234375, 0.34046875, 0.3475625, 0.350625, 0.348921875],
                                          [0.364953125, 0.363796875, 0.365078125, 0.36371875, 0.366234375, 0.363671875, 0.362375, 0.364203125, 0.3636875, 0.3635625],
                                          [0.356375, 0.350828125, 0.357640625, 0.355109375, 0.357546875, 0.355859375, 0.354109375, 0.353421875, 0.353921875, 0.355703125]])

    m = Metodo2kr(confidence=confidence, k=2, r=len(resultados[0]), r_results=resultados).metodo
    m.contra(c_q0=0, c_qa=1, c_qb=0, c_qab=-1, confidence=confidence)
    m.estimativa(c_q0=1, c_qa=-1, c_qb=1, c_qab=-1, confidence=confidence, m=10)

    print("------ Curvilinear")

    confidence = 0.95

    y = np.array([[0.339734375, 0.34215625, 0.3405625, 0.33984375, 0.342640625, 0.33996875, 0.34084375, 0.340890625, 0.340046875, 0.341515625],
        [0.356375, 0.350828125, 0.357640625, 0.355109375, 0.357546875, 0.355859375, 0.354109375, 0.353421875, 0.353921875, 0.355703125],
        [0.355953125, 0.34925, 0.345578125, 0.34434375, 0.349671875, 0.344015625, 0.345671875, 0.35090625, 0.351921875, 0.355640625],
        [0.34759375, 0.347953125, 0.349734375, 0.345453125, 0.351046875, 0.349921875, 0.353609375, 0.342625, 0.352015625, 0.349703125],
        [0.347109375, 0.34853125, 0.34278125, 0.336234375, 0.345, 0.339609375, 0.3425, 0.339078125, 0.348953125, 0.34628125],
        [0.346515625, 0.330921875, 0.34171875, 0.344875, 0.33796875, 0.339953125, 0.336875, 0.3465625, 0.342015625, 0.34571875],
        [0.337046875, 0.333625, 0.337640625, 0.335328125, 0.33840625, 0.335171875, 0.3364375, 0.33665625, 0.3363125, 0.340421875],
        [0.339375, 0.332984375, 0.338328125, 0.339953125, 0.336953125, 0.3378125, 0.3349375, 0.337859375, 0.341, 0.339875],
        [0.338875, 0.336609375, 0.33965625, 0.3375, 0.337546875, 0.3356875, 0.334421875, 0.3393125, 0.33896875, 0.337859375]])

    # regressao pronta
    y = y[1:]
    y = np.array([[i] for i in y.flatten()])
    x = np.array([i for i in range(2,10)])
    x = np.array([[i] *10  for i in x])
    x = np.array([[i] for i in x.flatten()])
    print("regressao linear sklearn")
    reg = LinearRegression().fit(x, y)
    print("score: ", reg.score(x, y))
    print("coef: ", reg.coef_)
    print("inter: ", reg.intercept_)
    #Curvilinear(confidence, x, y)

    # regressao linear padrao
    print("------- Regressao padrao")
    #y = np.mean(y[:, 1:], axis=1)
    y = y.flatten()
    x = x.flatten()
    # print("media: ", y)
    # print("x: ", x)
    r = RegressaoLinear(confidence=confidence, x=x, y=y, log=False)
    r.previsao(np.array([10, 11, 12, 13]), confidence)
    # y = y[1:].flatten()
    #y = np.mean(y, axis=1)
    #Curvilinear(confidence, x, y)

