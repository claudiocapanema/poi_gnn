import statistics as st
import math
import numpy as np

from .t_distribution import T_Distribution
import scipy.stats as stt

def t_distribution_test(x, confidence=0.95):

    n = len(x)
    decimals = 3
    mean = round(st.mean(x), decimals)
    liberty_graus = n
    s = st.stdev(x)
    alfa = 1 - confidence
    column = 1 - alfa / 2
    t_value = T_Distribution().find_t_distribution(column, liberty_graus)
    average_variation = round(t_value * (s / math.pow(n, 1/2)), decimals)
    average_variation = str(average_variation)
    while len(average_variation) < decimals + 2:
        average_variation = average_variation + "0"

    mean = str(mean)
    while len(mean) < decimals + 2:
        mean = mean + "0"

    ic = stt.t.interval(alpha=0.95, df=len(x) - 1, loc=np.mean(x), scale=stt.sem(x))
    l = round(ic[0], decimals)
    r = round(ic[1], decimals)
    library_variation = str(round(r - np.mean(x), decimals))
    #print("Library: ", library_variation, " local: ", average_variation)

    return str(mean) + u"\u00B1" + average_variation

def pmi(joint_frequency, n, frequency_left, frequency_right):

    cal = (joint_frequency * n)/(frequency_left * frequency_right)
    re = 0
    if cal > 0:
        re = math.log(cal)
        if math.isinf(re):
            re = 16664

    return re