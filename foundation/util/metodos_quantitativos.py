import math
import scipy
import statistics as st

import numpy as np
import pandas as pd
from scipy.stats import f_oneway
from t_distribution import T_Distribution

class UmFator:

    def __init__(self, confidence, results, decimals=6):
        # coluna é a variável
        self.t_distribution = T_Distribution()
        self.decimals = decimals
        self.confidence = confidence
        self.results = results
        self.a = self.results.shape[1]
        self.r = self.results.shape[0]
        self.medias = np.round(np.mean(results, axis=0), decimals)
        self.media_total = round(np.mean(self.medias), decimals)
        self.somas = np.round(np.sum(results, axis=0), decimals)
        self.soma_total = round(np.sum(self.somas), 2)
        self.efeitos = np.round(self.medias - self.media_total, decimals)

        self.sse = float
        self.ss0 = float
        self.ssa = float
        self.ssb = float
        self.ssc = float
        self.ssy = np.round(np.sum(np.power(self.results, 2)), decimals)

        self.msa = float
        self.mse = float

        self.se = float
        self.smu = float
        self.saj = float

        self.graus_de_liberdade = self.a * (self.r - 1)
        self.valor_t_ou_z = float
        self.ic_a = np.zeros(shape=(self.a, 2))
        self.ic_mu = np.zeros(shape=(2))

        self.dif_medias = np.zeros(shape=int(math.factorial(self.a)/(2*math.factorial(self.a - 2))))
        self.s_medias_alfas = float
        self.ic_dif_alfas = np.zeros(shape=(int(math.factorial(self.a)/(2*math.factorial(self.a - 2))), 2))

        self.run()
        #self.run_library()

    def run_library(self):

        print("Biblioteca")
        if self.results.shape[1] == 3:
            print(f_oneway(self.results[0], self.results[1], self.results[2]))
        elif self.results.shape[1] == 4:
            print(f_oneway(self.results[0], self.results[1], self.results[2], self.results[3]))


    def run(self):

        mean_matrix = np.\
            reshape([self.media_total]*
                    self.results.shape[0]*
                    self.results.shape[1], newshape=self.results.shape)
        efects_matrix = np.\
            reshape([list(self.efeitos)]*
                    self.results.shape[0], newshape=self.results.shape)
        erros_matrix = self.results - mean_matrix - efects_matrix
        print("Matriz de erros: ", self.results - mean_matrix)
        self.sse = round(np.sum(np.power(erros_matrix, 2)), self.decimals)

        self.ss0 = round(self.a*self.r*self.media_total*self.media_total, self.decimals)

        self.ssa = round(self.r*np.sum(np.power(self.efeitos, 2)), self.decimals)

        self.sst = round(self.ssy - self.ss0, self.decimals)

        self.msa = round(self.ssa/(self.a - 1), self.decimals)
        self.mse = round(self.sse / (self.a * (self.r - 1)), self.decimals)

        self.se = round(math.sqrt(self.mse), self.decimals)

        self.smu = round(self.se/(math.sqrt(self.a * self.r)), self.decimals)
        self.saj = round(self.se*(math.sqrt(((self.a -1)/(self.a * self.r)))), self.decimals)

        alfa = 1 - self.confidence
        column = 1 - alfa / 2

        if self.graus_de_liberdade < 30:

            self.valor_t_ou_z = self.t_distribution.find_t_distribution(column, self.graus_de_liberdade)

        ic_min = self.media_total - self.valor_t_ou_z * self.smu
        ic_max = self.media_total + self.valor_t_ou_z * self.smu
        self.ic_mu = np.round(np.array([ic_min, ic_max]), self.decimals)

        for i in range(self.ic_a.shape[0]):
            ic_min = self.efeitos[i] - self.valor_t_ou_z * self.saj
            ic_max = self.efeitos[i] + self.valor_t_ou_z * self.saj
            self.ic_a[i, 0] = ic_min
            self.ic_a[i, 1] = ic_max

        self.ic_a = np.round(self.ic_a, self.decimals)

        if self.a == 3:
            self.comparar_3_alfas()
        else:
            self.comparar_4_alfas()

        print("Somas: ", self.somas)
        print("Soma total: ", self.soma_total)
        print("Médias: ", self.medias)
        print("Média total: ", self.media_total)
        print("Efeitos: ", self.efeitos)

        print("Matrizes")
        print("matriz de resultados: ", self.results)
        print("matriz de média das médias: ", np.reshape(np.repeat(self.media_total,
                                                        repeats=self.results.shape[0]*self.results.shape[1]), newshape=(self.results.shape[0], self.results.shape[1])))

        matriz_medias = []
        for i in range(self.results.shape[0]):
            matriz_medias.append(self.efeitos)
        matriz_medias = np.array(matriz_medias)
        print("matriz de médias: ", matriz_medias)
        print("matriz de erros: ", erros_matrix)
        print("ssy: ", self.ssy)
        print("sst: ", self.sst)
        print("sse: ", self.sse)
        print("ss0: ", self.ss0)
        print("ssa: ", self.ssa)
        print("sst: ", self.sst)

        print("Variação explicada pelo fator: ", self.ssa/self.sst)

        print("MSA: ", self.msa)
        print("MSE: ", self.mse)
        print("MSA/MSE (F computado): ", self.msa/self.mse)
        print("pesquisar F tabela para uma confiança = 1 - alfa -> numerador: ", self.a - 1, " / denominador: ", (self.a*(self.r - 1)))
        print("Desvio padrão dos erros (se = raiz(MSE)): ", self.se)
        print("Desvio padrão de mu: ", self.smu)
        print("Desvio padrão alfa j (aj): ", self.saj)
        print("Comparação de alfas")
        print("Diferanças entre alfas: \n", self.dif_medias)
        print("Desvio padrão entre alfa: ", self.s_medias_alfas)
        print("Intervalo de confiança mu: ", self.ic_mu)
        print("Intervalo de confiança aj: ", self.ic_a)
        print("Confiança: ", self.confidence)
        print("Confiança (coluna): ", column)
        print("Graus de liberdade: ", self.graus_de_liberdade, " valor t: ", self.valor_t_ou_z)

        print("Intervalo de confiança comparação alfas (contraste): \n", self.ic_dif_alfas)

    def comparar_3_alfas(self):

        # comparando elementos
        self.dif_medias[0] = self.medias[0] - self.medias[1]
        self.dif_medias[1] = self.medias[0] - self.medias[2]
        self.dif_medias[2] = self.medias[1] - self.medias[2]

        self.s_medias_alfas = self.se*math.sqrt(2/self.r)
        ic_min = round(self.dif_medias[0] - self.valor_t_ou_z * self.s_medias_alfas, self.decimals)
        ic_max = round(self.dif_medias[0] + self.valor_t_ou_z * self.s_medias_alfas, self.decimals)
        self.ic_dif_alfas[0][0] = ic_min
        self.ic_dif_alfas[0][1] = ic_max
        ic_min = round(self.dif_medias[1] - self.valor_t_ou_z * self.s_medias_alfas, self.decimals)
        ic_max = round(self.dif_medias[1] + self.valor_t_ou_z * self.s_medias_alfas, self.decimals)
        self.ic_dif_alfas[1][0] = ic_min
        self.ic_dif_alfas[1][1] = ic_max
        ic_min = round(self.dif_medias[2] - self.valor_t_ou_z * self.s_medias_alfas, self.decimals)
        ic_max = round(self.dif_medias[2] + self.valor_t_ou_z * self.s_medias_alfas, self.decimals)
        self.ic_dif_alfas[2][0] = ic_min
        self.ic_dif_alfas[2][1] = ic_max

    def comparar_4_alfas(self):

        # comparando elementos
        self.dif_medias[0] = self.medias[0] - self.medias[1]
        self.dif_medias[1] = self.medias[0] - self.medias[2]
        self.dif_medias[2] = self.medias[0] - self.medias[3]
        self.dif_medias[3] = self.medias[1] - self.medias[2]
        self.dif_medias[4] = self.medias[1] - self.medias[3]
        self.dif_medias[5] = self.medias[2] - self.medias[3]

        self.s_medias_alfas = self.se*math.sqrt(2/self.r)
        ic_min = round(self.dif_medias[0] - self.valor_t_ou_z * self.s_medias_alfas, self.decimals)
        ic_max = round(self.dif_medias[0] + self.valor_t_ou_z * self.s_medias_alfas, self.decimals)
        self.ic_dif_alfas[0][0] = ic_min
        self.ic_dif_alfas[0][1] = ic_max
        ic_min = round(self.dif_medias[1] - self.valor_t_ou_z * self.s_medias_alfas, self.decimals)
        ic_max = round(self.dif_medias[1] + self.valor_t_ou_z * self.s_medias_alfas, self.decimals)
        self.ic_dif_alfas[1][0] = ic_min
        self.ic_dif_alfas[1][1] = ic_max
        ic_min = round(self.dif_medias[2] - self.valor_t_ou_z * self.s_medias_alfas, self.decimals)
        ic_max = round(self.dif_medias[2] + self.valor_t_ou_z * self.s_medias_alfas, self.decimals)
        self.ic_dif_alfas[2][0] = ic_min
        self.ic_dif_alfas[2][1] = ic_max

        ic_min = round(self.dif_medias[3] - self.valor_t_ou_z * self.s_medias_alfas, self.decimals)
        ic_max = round(self.dif_medias[3] + self.valor_t_ou_z * self.s_medias_alfas, self.decimals)
        self.ic_dif_alfas[3][0] = ic_min
        self.ic_dif_alfas[3][1] = ic_max

        ic_min = round(self.dif_medias[4] - self.valor_t_ou_z * self.s_medias_alfas, self.decimals)
        ic_max = round(self.dif_medias[4] + self.valor_t_ou_z * self.s_medias_alfas, self.decimals)
        self.ic_dif_alfas[4][0] = ic_min
        self.ic_dif_alfas[4][1] = ic_max

        ic_min = round(self.dif_medias[5] - self.valor_t_ou_z * self.s_medias_alfas, self.decimals)
        ic_max = round(self.dif_medias[5] + self.valor_t_ou_z * self.s_medias_alfas, self.decimals)
        self.ic_dif_alfas[5][0] = ic_min
        self.ic_dif_alfas[5][1] = ic_max


class Metodo22r:

    def __init__(self, confidence, r, r_results, decimals=6, log_y = False):
        self.decimals = decimals
        self.t_distribution = T_Distribution()
        self.confidence = confidence
        self.r=r
        if not log_y:
            self.r_results = r_results
        else:
            self.r_results = np.array([[math.log(j, 10) for j in i] for i in r_results])
            print("resultado: ", self.r_results)
        self.media_y = np.round(np.mean(self.r_results, axis=1), decimals)
        print("media y", self.media_y)
        self.erros = np.zeros(shape=(4, self.r))
        self.q0 = float
        self.qa = float
        self.qb = float
        self.qc = float

        self.porcentagem_fator_a = float
        self.porcentagem_fator_b = float
        self.porcentagem_fator_c = float
        self.porcentagem_fator_e = float

        self.sse = float
        self.ss0 = float
        self.ssa = float
        self.ssb = float
        self.ssc = float
        self.ssy = np.round(np.sum(np.power(self.r_results, 2)), decimals)
        self.table = pd.DataFrame({'i':[1,1,1,1], 'a':[-1,1,-1,1], 'b':[-1,-1,1,1], 'c':[1,-1,-1,1]})

        self.se = float
        self.si = float

        self.ica = [float, float]
        self.icb = [float, float]
        self.icc = [float, float]
        self.ic0 = [float, float]

        self.graus_de_liberdade = float
        self.valor_t_ou_z = float

        self.gerar_erros()
        self.gerar_efeitos()

    def gerar_erros(self):

        sse = 0
        for i in range(self.erros.shape[0]):
            for j in range(self.erros.shape[1]):
                self.erros[i][j] = self.r_results[i][j] - self.media_y[i]
                sse+= self.erros[i][j]*self.erros[i][j]

        self.sse = round(sse, self.decimals)

    def gerar_efeitos(self):
        q0 = 0
        qa = 0
        qb = 0
        qc = 0

        for i in range(self.table.shape[0]):

            q0 += round(self.table['i'].iloc[i]*self.media_y[i], self.decimals)
            qa += round(self.table['a'].iloc[i]*self.media_y[i], self.decimals)
            qb += round(self.table['b'].iloc[i]*self.media_y[i], self.decimals)
            qc += round(self.table['c'].iloc[i]*self.media_y[i], self.decimals)

        self.q0 = round(q0/self.table.shape[0], self.decimals)
        self.qa = round(qa/self.table.shape[0], self.decimals)
        self.qb = round(qb/self.table.shape[0], self.decimals)
        self.qc = round(qc/self.table.shape[0], self.decimals)

        print("Efeitos")
        print("q0: ", self.q0, " qa: ", self.qa, " qb: ", self.qb, " qc: ", self.qc)

        print("replicações: ", self.r)
        con = round(2*2*self.r, self.decimals)
        self.ss0 = round(con*self.q0*self.q0, self.decimals)
        self.ssa = round(con * self.qa * self.qa, self.decimals)
        self.ssb = round(con * self.qb * self.qb, self.decimals)
        self.ssc = round(con * self.qc * self.qc, self.decimals)

        self.sst = round(self.ssy - self.ss0, self.decimals)

        print("Erros: ")
        print("ss0: ", self.ss0, " ssa: ",
              self.ssa, " ssb: ", self.ssb,
              " ssc: ", self.ssc, " sse: ",
              self.sse, " ssy: ", self.ssy, " sst: ", self.sst)

        self.porcentagem_fator_a = round(self.ssa / self.sst, self.decimals)
        self.porcentagem_fator_b = round(self.ssb / self.sst, self.decimals)
        self.porcentagem_fator_c = round(self.ssc / self.sst, self.decimals)
        self.porcentagem_fator_e = round(self.sse / self.sst, self.decimals)

        print("Porcentagens fatores: ")
        print("A: ", self.porcentagem_fator_a, "B: ", self.porcentagem_fator_b, "c: ",
              self.porcentagem_fator_c, "e: ", self.porcentagem_fator_e)

        self.se = round(math.sqrt((self.sse/(2*2*(self.r -1)))), self.decimals)
        self.si = round(self.se/math.sqrt((2*2*self.r)), self.decimals)

        print("Desvio padrão dos erros (se): ", self.se)
        print("Desvio padrão dos efeitos (si): ", self.si)

        alfa = 1 - self.confidence
        column = 1 - alfa / 2

        self.graus_de_liberdade = 2*2*(self.r - 1)
        print("Confiança da tabela de : ", column)
        print("Graus de liberdade: ", self.graus_de_liberdade)

        if self.graus_de_liberdade <= 30:

            self.valor_t_ou_z = round(self.t_distribution.find_t_distribution(column, self.graus_de_liberdade), self.decimals)
        else:
            if column == 0.95:
                self.valor_t_ou_z = 1.645
            elif column == 0.975:
                self.valor_t_ou_z = 1.960
            elif column == 0.9:
                self.valor_t_ou_z = 1.282
            else:
                self.valor_t_ou_z = float(input("Digite z_value de "+str(column)+": "))

        print("valor t ou z: ", self.valor_t_ou_z)

        ic_min = round(self.qa - self.valor_t_ou_z*self.si, self.decimals)
        ic_max = round(self.qa + self.valor_t_ou_z*self.si, self.decimals)
        self.ica = [ic_min, ic_max]

        ic_min = round(self.qb - self.valor_t_ou_z * self.si, self.decimals)
        ic_max = round(self.qb + self.valor_t_ou_z * self.si, self.decimals)
        self.icb = [ic_min, ic_max]

        ic_min = round(self.qc - self.valor_t_ou_z * self.si, self.decimals)
        ic_max = round(self.qc + self.valor_t_ou_z * self.si, self.decimals)
        self.icc = [ic_min, ic_max]

        ic_min = round(self.q0 - self.valor_t_ou_z * self.si, self.decimals)
        ic_max = round(self.q0 + self.valor_t_ou_z * self.si, self.decimals)
        self.ic0 = [ic_min, ic_max]

        print("Intervalos de confiança dos efeitos para " + str(self.confidence) + " de confiança")
        print("IC_A: ", self.ica, "IC_B: ", self.icb, "IC_C: ", self.icc, "IC_0: ", self.ic0)

        self.ms = (self.ssa + self.ssb + self.ssc)/3
        self.mse = self.sse/self.graus_de_liberdade

        print("MS: ", self.ms)
        print("MSE: ", self.mse)
        print("F computado: ", self.ms/self.mse)
        print("F tabela (", self.confidence, ", 3, ", self.graus_de_liberdade, ")")

    def contra(self, c_q0, c_qa, c_qb, c_qab, confidence):


        u = c_q0 * self.q0 + c_qa * self.qa + c_qb * self.qb + c_qab * self.qc

        su2 = (pow(self.se, 2) * (c_q0 * c_q0 + c_qa * c_qa
                                  + c_qb * c_qb + c_qab * c_qab))\
              /(2*2*self.r)
        su = math.sqrt(su2)

        alfa = 1 - confidence
        column = 1 - alfa / 2

        print("/ Contraste /")
        print("Confiança da tabela de : ", column)
        print("Graus de liberdade: ", self.graus_de_liberdade)

        if self.graus_de_liberdade <= 30:

            valor_t_ou_z = round(self.t_distribution.find_t_distribution(column, self.graus_de_liberdade),
                                      self.decimals)
        else:
            if column == 0.95:
                valor_t_ou_z = 1.645
            elif column == 0.975:
                valor_t_ou_z = 1.960
            elif column == 0.9:
                valor_t_ou_z = 1.282
            else:
                valor_t_ou_z = float(input("Digite z_value de " + str(column) + ": "))

        print("valor t ou z: ", valor_t_ou_z)

        print("u: ", u)
        print("su2: ", su2)
        print("su: ", su)

        ic_min = round(u - valor_t_ou_z * su, self.decimals)
        ic_max = round(u + valor_t_ou_z * su, self.decimals)

        print("IC contraste: ", [ic_min, ic_max])


    def estimativa(self, c_q0, c_qa, c_qb, c_qab, confidence, m):


        u = c_q0 * self.q0 + c_qa * self.qa + c_qb * self.qb + c_qab * self.qc

        alfa = 1 - confidence
        column = 1 - alfa / 2

        print("/ Estimar /")
        print("Confiança da tabela de : ", column)
        print("Graus de liberdade: ", self.graus_de_liberdade)

        if self.graus_de_liberdade <= 30:

            valor_t_ou_z = round(self.t_distribution.find_t_distribution(column, self.graus_de_liberdade),
                                      self.decimals)
        else:
            if column == 0.95:
                valor_t_ou_z = 1.645
            elif column == 0.975:
                valor_t_ou_z = 1.960
            elif column == 0.9:
                valor_t_ou_z = 1.282
            else:
                valor_t_ou_z = float(input("Digite z_value de " + str(column) + ": "))

        print("valor t ou z: ", valor_t_ou_z)

        print("ym: ", u)

        nef = ((2*2*self.r)/(1 + 2*2))
        sym = self.se * math.sqrt((1/nef + 1/m))
        y1 = u

        print("neff: ", nef)
        print("sym: ", sym)

        ic_min = round(y1 - valor_t_ou_z*sym, self.decimals)
        ic_max = round(y1 + valor_t_ou_z * sym, self.decimals)

        print("IC resposta media: ", [ic_min, ic_max])




class Metodo2kr:

    def __init__(self, confidence, k, r, r_results):
        if k==2:
            self.metodo = Metodo22r(confidence, r, r_results)


class MetodosQuantitativos:

    def __init__(self):
        self.t_table = T_Distribution()

    def _intervalo_de_confianca_de_dois_lados_proporcao(self, p, confidence, n):

        alfa = 1 - confidence
        column = 1 - alfa / 2

        graus_de_liberdade = n - 1
        #graus_de_liberdade = 31
        # if graus_de_liberdade < 30:
        #     print("alfa: ", alfa, " 1 - alfa/2: ", column)
        #     print("graus de liberdade: ", graus_de_liberdade)
        #     valor_t_ou_z = self.t_table.find_t_distribution(column, graus_de_liberdade)
        #
        # else:
        if column == 0.95:
            valor_t_ou_z = 1.645
        elif column == 0.975:
            valor_t_ou_z = 1.960
        elif column == 0.9:
            valor_t_ou_z = 1.282
        else:
            valor_t_ou_z = float(input("Digite z_value de " + str(column) + ": "))

        print("valor t ou z: ", valor_t_ou_z)
        print("componente: ", valor_t_ou_z * math.sqrt(((p*(1 - p))/n)))
        ic_min = p - valor_t_ou_z * math.sqrt(((p*(1 - p))/n))
        ic_max = p + valor_t_ou_z * math.sqrt(((p * (1 - p)) / n))

        print("IC p: ", [ic_min, ic_max])

    def _intervalo_de_confianca_de_um_lado_proporcao(self, p, confidence, n, type):

        alfa = 1 - confidence
        column = 1 - alfa

        graus_de_liberdade = n - 1
        #graus_de_liberdade = 31
        # if graus_de_liberdade < 30:
        #     print("alfa: ", alfa, " 1 - alfa: ", column)
        #     print("graus de liberdade: ", graus_de_liberdade)
        #     valor_t_ou_z = self.t_table.find_t_distribution(column, graus_de_liberdade)
        #
        # else:
        if column == 0.95:
            valor_t_ou_z = 1.645
        elif column == 0.975:
            valor_t_ou_z = 1.960
        elif column == 0.9:
            valor_t_ou_z = 1.282
        else:
            valor_t_ou_z = float(input("Digite z_value de " + str(column) + ": "))

        print("Confiança: ", confidence)
        print("valor t ou z: ", valor_t_ou_z)
        print("componente: ", valor_t_ou_z * math.sqrt(((p*(1 - p))/n)))
        if type == "inferior":
             result = [p - valor_t_ou_z * math.sqrt(((p*(1 - p))/n)), "+inf"]
        elif type == "superior":
            result = ["-inf", p + valor_t_ou_z * math.sqrt(((p * (1 - p)) / n))]

        print("tipo: ", type)
        print("IC p: ", result)

    def observacoes_pareadas_intervalo_de_confianca_de_um_lado(self, confidence, type, a=None, b=None, n_a=None, x=None, s=None, decimals=10):

        if a is not None and b is not None:
            n_a = len(a)
            n_b = len(b)
            if n_a != n_b:
                return "erro", "erro"

            a = np.array(a)
            b = np.array(b)
            dif = np.round(np.subtract(a, b), decimals)
            x = np.round(np.mean(dif), decimals)
            s = np.round(st.stdev(dif), decimals)
            print("Vetor diferenças: ", dif)

        print("Media diferenças: ", x)
        print("Desvio diferenças: ", s)

        alfa = 1 - confidence
        column = 1 - alfa

        graus_de_liberdade = n_a - 1
        print("Graus de liberdade: ", graus_de_liberdade)

        if graus_de_liberdade < 30:
            print("alfa: ", alfa, " 1 - alfa/2: ", column)
            print("graus de liberdade: ", graus_de_liberdade)
            valor_t_ou_z = self.t_table.find_t_distribution(column, graus_de_liberdade)

        else:
            if column == 0.95:
                valor_t_ou_z = 1.645
            elif column == 0.975:
                valor_t_ou_z = 1.960
            elif column == 0.9:
                valor_t_ou_z = 1.282
            else:
                valor_t_ou_z = float(input("Digite z_value de " + str(column) + ": "))

        print("Confianca: ", confidence)
        print("Confianca tabela: ", column)
        print("valor t ou z: ", valor_t_ou_z)

        if type == "inferior":
            result =  [x - valor_t_ou_z * (s / np.sqrt(n_a)), "+inf"]
        elif type == "superior":
            result = ["-inf", x + valor_t_ou_z * (s / np.sqrt(n_a))]

        print("type: ", type)
        print("IC: ", result)

    def observacoes_pareadas(self, confidence, a=None, b=None, n_a=None, x=None, s=None, decimals=6):


        if a is not None and b is not None:
            n_a = len(a)
            n_b = len(b)
            if n_a != n_b:
                return "erro", "erro"

            a = np.array(a)
            b = np.array(b)
            dif = np.round(np.subtract(a, b), decimals)
            x = np.round(np.mean(dif), decimals)
            s = np.round(st.stdev(dif), decimals)
            print("Vetor diferenças: ", dif)

        print("Media diferenças: ", x)
        print("Desvio diferenças: ", s)

        alfa = 1 - confidence
        column = 1 - alfa / 2

        graus_de_liberdade = n_a-1
        print("Graus de liberdade: ", graus_de_liberdade)

        if graus_de_liberdade < 30:
            print("alfa: ", alfa, " 1 - alfa/2: ", column)
            print("graus de liberdade: ", graus_de_liberdade)
            valor_t_ou_z = self.t_table.find_t_distribution(column, graus_de_liberdade)

        else:
            if column == 0.95:
                valor_t_ou_z = 1.645
            elif column == 0.975:
                valor_t_ou_z = 1.960
            elif column == 0.9:
                valor_t_ou_z = 1.282
            else:
                valor_t_ou_z = float(input("Digite z_value de " + str(column) + ": "))

        print("Confianca: ", confidence)
        print("Confianca tabela: ", column)
        print("valor t ou z: ", valor_t_ou_z)


        ic_min = x - valor_t_ou_z*(s/np.sqrt(n_a))
        ic_max = x + valor_t_ou_z*(s/np.sqrt(n_a))

        print("IC: ", [ic_min, ic_max])

        return ic_min, ic_max

    def _intervalo_de_confianca_de_dois_lados(self, a, confidence):

        alfa = 1 - confidence
        column = 1 - alfa / 2

        # conferir
        graus_de_liberdade = len(a) -  1
        if graus_de_liberdade < 30:
            print("alfa: ", alfa, " 1 - alfa/2: ", column)
            print("graus de liberdade: ", graus_de_liberdade)
            valor_t_ou_z = self.t_table.find_t_distribution(column, graus_de_liberdade)

        else:
            if column == 0.95:
                valor_t_ou_z = 1.645
            elif column == 0.975:
                valor_t_ou_z = 1.960
            elif column == 0.9:
                valor_t_ou_z = 1.282
            else:
                valor_t_ou_z = float(input("Digite z_value de " + str(column) + ": "))

        print("valor t ou z: ", valor_t_ou_z)

        s = st.stdev(a)
        raiz_n = np.sqrt(len(a))
        media = np.mean(a)
        coef = s/media

        print("Media: ", media)
        print("Desvio s: ", s)
        print("Raiz(n): ", raiz_n)
        print("Coeficiente de variação: ", coef)

        ic_min = media - valor_t_ou_z*(s/raiz_n)
        ic_max = media + valor_t_ou_z*(s/raiz_n)

        print("IC : ", [ic_min, ic_max])

        return  ic_min, ic_max, media



    def observacoes_nao_pareadas(self, a, b, confidence):

        print("/ Sistema a /")
        ic_min_a, ic_max_a, media_a = self._intervalo_de_confianca_de_dois_lados(a, confidence)
        print("IC a: ", [ic_min_a, ic_max_a])

        print("/ Sistema b /")
        ic_min_b, ic_max_b, media_b = self._intervalo_de_confianca_de_dois_lados(b, confidence)
        print("IC b: ", [ic_min_b, ic_max_b])

        if (media_a <= ic_max_b and media_a >= ic_min_b) or \
                (media_b <= ic_max_a and media_b >= ic_min_b):
            print("Existe interseção a nível de média entre os interlados \n Fazer o teste-t")
        elif (ic_max_a < ic_min_b or ic_max_b < ic_min_a):
            print("Os sistemas são diferentes")
        else:
            print("Os sistemas não são diferentes")

    def tamanho_do_modelo_para_erro_maximo(self, erro, confidence, a=None, media_a=None, desvio_a=None, n_a=None):
        """

        :param erro: 5 é 5%, 10 é 10%.
        :param confidence:
        :param a:
        :param media_a:
        :param desvio_a:
        :param n_a:
        :return:
        """


        alfa = 1 - confidence
        column = 1 - alfa / 2
        if a is not None:
            media_a = st.mean(a)
            desvio_a = st.stdev(a)
            n_a = len(a)
        elif not (media_a is not None and desvio_a is not None and n_a is not None):
            print("faltam parametros")
            return
        graus_de_liberdade = n_a - 1
        if graus_de_liberdade < 30:
            print("alfa: ", alfa, " 1 - alfa/2: ", column)
            print("graus de liberdade: ", graus_de_liberdade)
            valor_t_ou_z = self.t_table.find_t_distribution(column, graus_de_liberdade)

        else:
            if column == 0.95:
                valor_t_ou_z = 1.645
            elif column == 0.975:
                valor_t_ou_z = 1.960
            elif column == 0.9:
                valor_t_ou_z = 1.282
            else:
                valor_t_ou_z = float(input("Digite z_value de " + str(column) + ": "))

        print("valor t ou z: ", valor_t_ou_z)
        print("Média: ", media_a)
        print("Desvio: ", desvio_a)
        print("Erro: ", erro)
        print("Graus de liberdade: ", graus_de_liberdade)

        n = pow(((100 * valor_t_ou_z * desvio_a)/(erro*media_a)), 2)

        print("Quantidade de replicações para um erro de: ", n, " (", math.ceil(n), ")")

    def tamanho_do_modelo_para_erro_maximo_para_amostra(self, erro, confidence, p, n_original, r):
        """

        :param erro: 5 é 5%, 10 é 10%.
        :param confidence:
        :param a:
        :param media_a:
        :param desvio_a:
        :param n_a:
        :return:
        """


        alfa = 1 - confidence
        column = 1 - alfa / 2
        graus_de_liberdade = n_original - 1
        if graus_de_liberdade < 30:
            print("alfa: ", alfa, " 1 - alfa/2: ", column)
            print("graus de liberdade: ", graus_de_liberdade)
            valor_t_ou_z = self.t_table.find_t_distribution(column, graus_de_liberdade)

        else:
            if column == 0.95:
                valor_t_ou_z = 1.645
            elif column == 0.975:
                valor_t_ou_z = 1.960
            elif column == 0.9:
                valor_t_ou_z = 1.282
            else:
                valor_t_ou_z = float(input("Digite z_value de " + str(column) + ": "))

        print("valor t ou z: ", valor_t_ou_z)
        print("P: ", p)
        print("r: ", r)
        print("Erro: ", erro)
        print("Graus de liberdade: ", graus_de_liberdade)

        n = pow(valor_t_ou_z, 2) * (p*(1 - p))/pow(r, 2)

        print("Quantidade de replicações para um erro de: ", n, " (", math.ceil(n), ")")

    def _intervalo_de_confianca_de_um_lado(self, confidence, type, a=None, media=None, s=None, raiz_n=None, coef=None):

        alfa = 1 - confidence
        column = 1 - alfa

        # conferir
        graus_de_liberdade = len(a) - 1
        if graus_de_liberdade < 30:
            print("alfa: ", alfa, " 1 - alfa/2: ", column)
            print("graus de liberdade: ", graus_de_liberdade)
            valor_t_ou_z = self.t_table.find_t_distribution(column, graus_de_liberdade)

        else:
            if column == 0.95:
                valor_t_ou_z = 1.645
            elif column == 0.975:
                valor_t_ou_z = 1.960
            elif column == 0.9:
                valor_t_ou_z = 1.282
            else:
                valor_t_ou_z = float(input("Digite z_value de " + str(column) + ": "))

        print("valor t ou z: ", valor_t_ou_z)

        if a is not None:
            s = st.stdev(a)
            raiz_n = np.sqrt(len(a))
            media = np.mean(a)
            coef = s/media

        print("Media: ", media)
        print("Desvio s: ", s)
        print("Raiz(n): ", raiz_n)
        print("Coeficiente de variação: ", coef)

        ic_min = media - valor_t_ou_z*(s/raiz_n)
        ic_max = media + valor_t_ou_z*(s/raiz_n)

        if type == 'inferior':
            return [ic_min, '+inf']
        else:
            return ['-inf', ic_max]

    def limite_inferior(self, confidence, n_a=None,
                        n_b=None, x_a=None, x_b=None,
                        variance_a=None, variance_b=None,
                        a=None, b=None):

        return self.t_test(confidence=confidence,
                           n_a=n_a,
                           n_b=n_b,
                           x_a=x_a,
                           x_b=x_b,
                           variance_a=variance_a,
                           variance_b=variance_b,
                           a=a,
                           b=b,
                           test_type="limite_inferior")

    def limite_superior(self, confidence, n_a=None,
                        n_b=None, media_a=None, media_b=None,
                        variance_a=None, variance_b=None,
                        a=None, b=None, media_dif=None):
        if  (a is None and b is None and n_a is not None and n_a != n_b):
            return self.t_test(confidence=confidence,
                               n_a=n_a,
                               n_b=n_b,
                               x_a=media_a,
                               x_b=media_b,
                               variance_a=variance_a,
                               variance_b=variance_b,
                               a=a,
                               b=b,
                               test_type="limite_superior")
        elif (a is not None and b is not None):
            if len(a) != len(b):
                return self.t_test(confidence=confidence,
                                   n_a=n_a,
                                   n_b=n_b,
                                   x_a=media_a,
                                   x_b=media_b,
                                   variance_a=variance_a,
                                   variance_b=variance_b,
                                   a=a,
                                   b=b,
                                   test_type="limite_superior")
            else:
                self._intervalo_de_confianca_de_um_lado(confidence, type, a=None, media=None, s=None, raiz_n=None)

    def t_test_core(self, n_a, n_b, x_a, x_b, variance_a, variance_b):

        x_diff = x_a - x_b

        s_diff = math.sqrt((variance_a / n_a) + variance_b / n_b)

        v_num = pow((variance_a / n_a) + variance_b / n_b, 2)
        v_den_a = ((1 / (n_a - 1) * pow(variance_a / n_a, 2)))
        v_den_b = ((1 / (n_b - 1) * pow(variance_b / n_b, 2)))

        v = (v_num / (v_den_a + v_den_b)) - 2
        print("Quantil original: ", v)
        v = math.floor(v)

        return x_diff, s_diff, v

    def find_t_distribution(self, column, v):

        if column == 0.6:
            quantile = self.t_table['60'].iloc[v]
        elif column == 0.7:
            quantile = self.t_table['70'].iloc[v]
        elif column == 0.8:
            quantile = self.t_table['80'].iloc[v]
        elif column == 0.9:
            quantile = self.t_table['90'].iloc[v]
        elif column == 0.95:
            quantile = self.t_table['95'].iloc[v]
        elif column == 0.975:
            quantile = self.t_table['975'].iloc[v]
        elif column == 0.995:
            quantile = self.t_table['995'].iloc[v]
        elif column == 0.9995:
            quantile = self.t_table['9995'].iloc[v]

        return quantile

    def t_test(self, confidence, n_a=None, n_b=None, x_a=None, x_b=None,
               variance_a=None, variance_b=None, a=None, b=None,
               test_type="padrao"):

        """


        :param confidence:
        :param n_a:
        :param n_b:
        :param x_a:
        :param x_b:
        :param variance_a:
        :param variance_b:
        :param a:
        :param b:
        :param test_type: default "padrao".
        Other values are: "limite_inferior" and "limite_superior"
        :return:
        """

        if a is not None and b is not None:
            n_a = len(a)
            n_b = len(b)
            if n_a != n_b:
                print("Tamanhos diferentes")

            x_a = st.mean(a)
            x_b = st.mean(b)
            print("Média a: ", x_a)
            print("Média b: ", x_b)
            s_a = st.stdev(a)
            s_b = st.stdev(b)

            variance_a = st.variance(a)
            variance_b = st.variance(b)

            x_diff, s_diff, v = self.t_test_core(n_a, n_b, x_a, x_b, variance_a, variance_b)

        elif n_a is not None and n_b is not None\
                and x_a is not None and x_b is not None\
                and variance_a is not None\
                and variance_b is not None:

            x_diff, s_diff, v = self.t_test_core(n_a, n_b, x_a, x_b, variance_a, variance_b)
        else:
            return "erro", "erro"

        #x_diff, s_diff, v = self.t_test_core(n_a, n_b, x_a, x_b, variance_a, variance_b)

        alfa = 1 - confidence
        if test_type == "padrao":
            column = 1-alfa/2
        else:
            column = confidence

        if v <= 30:
            # if test_type=="limite_inferior":
            #     v+=-1
            valor_t_ou_z = self.t_table.find_t_distribution(column, v)
        else:
            if column == 0.95:
                valor_t_ou_z = 1.645
            elif column == 0.975:
                valor_t_ou_z = 1.960
            elif column == 0.9:
                valor_t_ou_z = 1.282
            else:
                valor_t_ou_z = float(input("Digite z_value de " + str(column) + ": "))

        print("Variancia a (S^2a): ", variance_a)
        print("Variancia b (S^2b): ", variance_b)
        print("n a: ", n_a)
        print("n b: ", n_b)
        print("Diferença das médias: ", x_diff)
        print("Desvio das diferença média (Sd): ", s_diff)
        print("Graus de liberdade: ", v)
        print("Confiança: ", confidence)
        print("Confiança na tabela: ", column)
        print("Valor z ou t: ", valor_t_ou_z)

        if test_type == "padrao":
            ic_min = x_diff-valor_t_ou_z*s_diff
            ic_max = x_diff+valor_t_ou_z*s_diff
        elif test_type == "inferior":
            ic_min = x_diff - valor_t_ou_z * s_diff
            ic_max = "inf+"
        elif test_type == "superior":
            ic_min = "-inf"
            ic_max = x_diff + valor_t_ou_z * s_diff

        print("IC teste-t (dif_medias +- valor_t_ou_z * Sd)", [ic_min, ic_max])

        return ic_min, ic_max


# class T_Distribution:
#
#     def __init__(self):
#
#         self.q_60 = [0.325,
#                     0.289,
#                     0.277,
#                     0.271,
#                     0.267,
#                     0.265,
#                     0.263,
#                     0.262,
#                     0.261,
#                     0.260,
#                     0.260,
#                     0.259,
#                     0.259,
#                     0.258,
#                     0.258,
#                     0.258,
#                     0.257,
#                     0.257,
#                     0.257,
#                     0.257,
#                     0.257,
#                     0.256,
#                     0.256,
#                     0.256,
#                     0.256,
#                     0.256,
#                     0.256,
#                     0.256,
#                     0.256,
#                     0.256,
#                     0.254,
#                     0.254,
#                     0.254]
#
#         self.q_70 = [0.727,
#                     0.617,
#                     0.584,
#                     0.569,
#                     0.559,
#                     0.553,
#                     0.549,
#                     0.546,
#                     0.543,
#                     0.542,
#                     0.540,
#                     0.539,
#                     0.538,
#                     0.537,
#                     0.536,
#                     0.535,
#                     0.534,
#                     0.534,
#                     0.533,
#                     0.533,
#                     0.532,
#                     0.532,
#                     0.532,
#                     0.531,
#                     0.531,
#                     0.531,
#                     0.531,
#                     0.530,
#                     0.530,
#                     0.530,
#                     0.527,
#                     0.526,
#                     0.526]
#
#         self.q_80 = [1.377,
#                     1.061,
#                     0.978,
#                     0.941,
#                     0.920,
#                     0.906,
#                     0.896,
#                     0.889,
#                     0.883,
#                     0.879,
#                     0.876,
#                     0.873,
#                     0.870,
#                     0.868,
#                     0.866,
#                     0.865,
#                     0.863,
#                     0.862,
#                     0.861,
#                     0.860,
#                     0.859,
#                     0.858,
#                     0.858,
#                     0.857,
#                     0.856,
#                     0.856,
#                     0.855,
#                     0.855,
#                     0.854,
#                     0.854,
#                     0.848,
#                     0.846,
#                     0.845]
#
#         self.q_90 = [3.078,
#                         1.886,
#                         1.638,
#                         1.533,
#                         1.476,
#                         1.440,
#                         1.415,
#                         1.397,
#                         1.383,
#                         1.372,
#                         1.363,
#                         1.356,
#                         1.350,
#                         1.345,
#                         1.341,
#                         1.337,
#                         1.333,
#                         1.330,
#                         1.328,
#                         1.325,
#                         1.323,
#                         1.321,
#                         1.319,
#                         1.318,
#                         1.316,
#                         1.315,
#                         1.314,
#                         1.313,
#                         1.311,
#                         1.310,
#                         1.296,
#                         1.291,
#                         1.289]
#
#         self.q_95 = [6.314,
#                     2.920,
#                     2.353,
#                     2.132,
#                     2.015,
#                     1.943,
#                     1.895,
#                     1.860,
#                     1.833,
#                     1.812,
#                     1.796,
#                     1.782,
#                     1.771,
#                     1.761,
#                     1.753,
#                     1.746,
#                     1.740,
#                     1.734,
#                     1.729,
#                     1.725,
#                     1.721,
#                     1.717,
#                     1.714,
#                     1.711,
#                     1.708,
#                     1.706,
#                     1.703,
#                     1.701,
#                     1.699,
#                     1.697,
#                     1.671,
#                     1.662,
#                     1.658]
#
#         self.q_975 = [12.706,
#                     4.303,
#                     3.182,
#                     2.776,
#                     2.571,
#                     2.447,
#                     2.365,
#                     2.306,
#                     2.262,
#                     2.228,
#                     2.201,
#                     2.179,
#                     2.160,
#                     2.145,
#                     2.131,
#                     2.120,
#                     2.110,
#                     2.101,
#                     2.093,
#                     2.086,
#                     2.080,
#                     2.074,
#                     2.069,
#                     2.064,
#                     2.060,
#                     2.056,
#                     2.052,
#                     2.048,
#                     2.045,
#                     2.042,
#                     2.000,
#                     1.987,
#                     1.980]
#
#         self.q_995 = [63.657,
#                     9.925,
#                     5.841,
#                     4.604,
#                     4.032,
#                     3.707,
#                     3.499,
#                     3.355,
#                     3.250,
#                     3.169,
#                     3.106,
#                     3.055,
#                     3.012,
#                     2.977,
#                     2.947,
#                     2.921,
#                     2.898,
#                     2.878,
#                     2.861,
#                     2.845,
#                     2.831,
#                     2.819,
#                     2.807,
#                     2.797,
#                     2.787,
#                     2.779,
#                     2.771,
#                     2.763,
#                     2.756,
#                     2.750,
#                     2.660,
#                     2.632,
#                     2.617]
#
#         self.q_9995 = [636.619,
#                     31.599,
#                     12.924,
#                     8.610,
#                     6.869,
#                     5.959,
#                     5.408,
#                     5.041,
#                     4.781,
#                     4.587,
#                     4.437,
#                     4.318,
#                     4.221,
#                     4.140,
#                     4.073,
#                     4.015,
#                     3.965,
#                     3.922,
#                     3.883,
#                     3.850,
#                     3.819,
#                     3.792,
#                     3.768,
#                     3.745,
#                     3.725,
#                     3.707,
#                     3.690,
#                     3.674,
#                     3.659,
#                     3.646,
#                     3.460,
#                     3.402,
#                     3.373]
#
#         self.t_table = pd.DataFrame({'60': self.q_60, '70': self.q_70,
#                                      '80': self.q_80, '90': self.q_90,
#                                      '95': self.q_95, '975': self.q_975,
#                                      '995': self.q_995, '9995': self.q_9995})
#
#     def find_t_distribution(self, column, v):
#
#         v-=1
#
#         if column == 0.6:
#             quantile = self.t_table['60'].iloc[v]
#         elif column == 0.7:
#             quantile = self.t_table['70'].iloc[v]
#         elif column == 0.8:
#             quantile = self.t_table['80'].iloc[v]
#         elif column == 0.9:
#             quantile = self.t_table['90'].iloc[v]
#         elif column == 0.95:
#             quantile = self.t_table['95'].iloc[v]
#         elif column == 0.975:
#             quantile = self.t_table['975'].iloc[v]
#         elif column == 0.995:
#             quantile = self.t_table['995'].iloc[v]
#         elif column == 0.9995:
#             quantile = self.t_table['9995'].iloc[v]
#
#         return quantile


class Z_Distribution:

    def __init__(self):
        pass
