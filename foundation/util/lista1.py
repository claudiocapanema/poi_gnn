import numpy as np

from metodos_quantitativos import MetodosQuantitativos, Metodo2kr, UmFator

if __name__ == "__main__":

    m = MetodosQuantitativos()

    print("\n------ 6 a)")
    a = [17, 12, 9, 11, 14, 12]
    b = [20, 6, 10, 12, 15, 7, 9, 10]

    confidence = 0.9

    m.observacoes_nao_pareadas(a=a, b=b, confidence=confidence)

    print("\n----- 6 b)")
    a = [17, 12, 9, 11, 14, 12]
    b = [20, 6, 10, 12, 15, 7, 9, 10]

    confidence = 0.95

    m.t_test(confidence=confidence, a=a, b=b)

    print("\n----- 6 c)")
    a = [17, 12, 9, 11, 14, 12]

    confidence = 0.95
    erro = 10
    m.tamanho_do_modelo_para_erro_maximo(erro=erro, confidence=confidence, a=a)

    # print("\n----- tamanho proporcao")
    # confidence = 0.9
    # p = 0.5
    # m.tamanho_do_modelo_para_erro_maximo_para_amostra(erro, confidence, p, n_original, r)

    print("\n---- 6 d)")

    p = 0.333
    confidence = 0.95
    n = 6
    m._intervalo_de_confianca_de_um_lado_proporcao(p, confidence, n, "superior")

    print("\n---- 6 e)")

    confidence = 0.95
    a = [17, 12, 9, 11, 14, 12]
    b = [21, 18, 8, 13, 17, 17]

    m.observacoes_pareadas(confidence=confidence, a=a, b=b)
    
    print("\n---- 6 f)")
    
    confidence = 0.975

    a = [17, 12, 9, 11, 14, 12]
    b = [21, 18, 8, 13, 17, 17]

    m.observacoes_pareadas_intervalo_de_confianca_de_um_lado(a=a, b=b, confidence=confidence, type='superior', decimals=6)

    # 7 a e b
    # fazer manualmente
    print("\n---- 7 c)")
    a = [70, 74, 64, 68, 72, 78, 71, 64]
    m.tamanho_do_modelo_para_erro_maximo(erro=1, confidence=0.95, a=a)

    print("\n--- 7 d)")
    confidence = 0.6
    x_a = 70.125
    variance_a = 4.7939*4.7939
    n_a = 8
    x_b = 72
    variance_b = 3.6*3.6
    n_b = 10

    m.t_test(confidence=confidence, n_a=n_a, n_b=n_b, x_a=x_a, x_b=x_b,
               variance_a=variance_a, variance_b=variance_b)

    print("\n---- 7 e)")
    confidence = 0.8
    m.t_test(confidence=confidence, n_a=n_a, n_b=n_b, x_a=x_a, x_b=x_b,
             variance_a=variance_a, variance_b=variance_b, test_type="superior")

    print("\n---- 7 f)")
    # pegadinha
    confidence = 0.95
    p = 2/8
    n = 8
    m._intervalo_de_confianca_de_dois_lados_proporcao(confidence=confidence, p=p, n=n)

    print("\n---- 8 a)")
    a = [70,80,69,89,65,30,80,82,65,45]
    b = [80,85,70,90,60,32,89,80,70,40]
    m.observacoes_pareadas(confidence=confidence, a=a, b=b)

    print("\n---- 8 b)")
    confidence = 0.95
    p = 7/10
    n = 10
    m._intervalo_de_confianca_de_um_lado_proporcao(p, confidence, n, type="inferior")