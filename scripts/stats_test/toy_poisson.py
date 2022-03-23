import sys
import math
import random
import statistics
import scipy
import scipy.stats
import sympy
import numpy
from matplotlib import pyplot as plt

def make_basic_poisson_plots():
    def log_poisson(n,v):
        lp = numpy.full(len(n),-v).astype(float)
        nz = n!=0
        n = n[nz]
        lp[nz] = n - v + n*numpy.log(v/n) - (1/2)*numpy.log(2*math.pi*n)
        return lp
    

    expectation = 60
    observed = 52
    max_n = int(expectation)*2
    poisson_inputs = numpy.arange(0,max_n,1)
    log_poisson_values = log_poisson(poisson_inputs,expectation)
    poisson_values = numpy.exp(log_poisson_values)
    cumulative_poisson = poisson_values.cumsum()
    pvalue = cumulative_poisson[observed]

    fig, ax = plt.subplots()
    ax.plot(poisson_inputs, poisson_values, label='Poisson PDF')
    ax.axvline(observed, ls='--', label=f'Observed n={observed}', color='red')
    ax.fill_between(range(0,observed+1), 0, poisson_values[:observed+1], color='blue', hatch='///', alpha=0.3, label=f'p-value={pvalue:.2f}')
    plt.xlabel('Number of Events')
    plt.ylabel('Probability')
    #plt.xlim(expectation, expectation*1.1)
    plt.ylim(0)
    ax.legend()
    plt.savefig('out/toy_poisson.pdf')
    plt.close()

    fig, ax = plt.subplots()
    ax.plot(poisson_inputs, cumulative_poisson, label='Poisson Cumulative PDF')
    ax.axvline(observed, ls='--', label=f'Observed n={observed}', color='red')
    ax.axhline(pvalue, ls='dotted', label=f'p-value={pvalue:.2f}', color='green')
    plt.xlabel('Number of Events')
    plt.ylabel('Cumulative Probability')
    #plt.xlim(expectation*.9, expectation*1.1)
    plt.ylim(0)
    ax.legend()
    plt.savefig('out/toy_Cpoisson.pdf')
    plt.close()



make_basic_poisson_plots()
