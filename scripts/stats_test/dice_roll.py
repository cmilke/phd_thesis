#!/usr/bin/python
import sys
import random
import math
import statistics
import sympy
import numpy
import scipy.stats
from matplotlib import pyplot as plt

_die_faces = 6
_biased_num = 1

def calculate_max_bias(vals=None, bgd_est=None, bins=None, prob=None):
    # Get probability function vector
    bias = sympy.Symbol('b')

    signal_total_P = _die_faces+bias-1
    expec_sig_vals = numpy.array([ 1 if i != _biased_num else bias for i in range(_die_faces) ])
    expec_sig_vals = expec_sig_vals/signal_total_P * (vals.sum()-bgd_est.sum())
    expec_vals = expec_sig_vals+bgd_est

    theoretical_pdf = expec_vals / expec_vals.sum()
    probability_vector = [ prob(vals.sum(), v, p) for v,p in zip(vals, theoretical_pdf) ]

    # Get log likelihood
    variadic_loglikelihood = -sympy.log(numpy.product(probability_vector))
    #print([ f'{n}: {float(variadic_loglikelihood.subs(bias,n)):.02f}' for n in numpy.arange(1,5,0.5) ])

    # Take derivative of log likelihood
    variadic_derivative = sympy.diff(variadic_loglikelihood, bias)

    # Find zero of derivative
    zeros = []
    for guess in numpy.arange(1,6,0.5):
        try:
            zero = sympy.solvers.solvers.nsolve(variadic_derivative, bias, guess)
            zeros.append(float(zero))
        except ValueError: pass
    if len(zeros) == 0: return None
    zero = statistics.mean(zeros)
    return zero


def evaluate_max_likelihood(vals=None, bgd_est=None, bins=None, name_prefix=''):
    # Define probability function
    probability_function = lambda N,n,p: sympy.functions.combinatorial.factorials.binomial(N,n) * p**n * (1-p)**(N-n)
    maximum_bias = calculate_max_bias(vals=vals, bgd_est=bgd_est, bins=bins, prob=probability_function)
    print(f'Bias Estimate = {maximum_bias:.02f}')

    #signal_total_P = _die_faces+maximum_bias-1
    #expec_sig_vals = numpy.array([ 1 if i != _biased_num else maximum_bias for i in range(_die_faces) ])
    #expec_sig_vals = expec_sig_vals/signal_total_P * vals.sum()
    #expec_vals = expec_sig_vals+bgd_est

    #total_events = vals.sum()
    #num_toys = 100
    #toy_bias_list = []
    #for i in range(num_toys):
    #    toy_distro = numpy.random.choice(bins[:-1], size=total_events, p=expec_vals/expec_vals.sum())
    #    hist, _ = numpy.histogram(toy_distro, bins=bins)
    #    toy_bias = calculate_max_bias(vals=hist, bgd_est=bgd_est, bins=bins, prob=probability_function)
    #    if toy_bias is not None and toy_bias > 0:
    #        toy_bias_list.append(toy_bias)
    #    if i%10==0: print(f'{i}', end=' ', flush=True)
    #print()
    #bias_avg = statistics.mean(toy_bias_list)
    #bias_std = statistics.stdev(toy_bias_list)

    #fig, ax = plt.subplots()
    #ax.hist(toy_bias_list)
    #ax.axvline(x=maximum_bias, color='purple')
    #ax.axvline(x=bias_avg+0*bias_std, color='blue', ls='--')
    #ax.axvline(x=bias_avg+1*bias_std, color='cyan', ls='--')
    #ax.axvline(x=bias_avg+2*bias_std, color='green', ls='--')
    #ax.axvline(x=bias_avg+3*bias_std, color='yellow', ls='--')
    #ax.axvline(x=bias_avg+4*bias_std, color='red', ls='--')
    #ax.axvline(x=bias_avg+5*bias_std, color='maroon', ls='--')
    #ax.axvline(x=bias_avg-1*bias_std, color='cyan', ls='--')
    #ax.axvline(x=bias_avg-2*bias_std, color='green', ls='--')
    #ax.axvline(x=bias_avg-3*bias_std, color='yellow', ls='--')
    #ax.axvline(x=bias_avg-4*bias_std, color='red', ls='--')
    #ax.axvline(x=bias_avg-5*bias_std, color='maroon', ls='--')
    #print(f'Bias Estimate = {maximum_bias:.02f}+-{bias_std:.02f} ({100*bias_std/maximum_bias:.1f}% Error)')

    #plt.savefig(name_prefix+'_biasEstimate.pdf')
    return maximum_bias


def draw_max_likelihood(vals=None, expec_vals=None, predicted_bias=None, yrange=None, bins=None, name_prefix=''):
    total_events = vals.sum()
    num_toys = 1000
    toy_hists = []
    for i in range(num_toys):
        toy_distro = numpy.random.choice(bins[:-1], size=total_events, p=expec_vals/expec_vals.sum())
        hist, _ = numpy.histogram(toy_distro, bins=bins)
        toy_hists.append(hist)
    toy_hists = numpy.array(toy_hists)

    toy_averages = numpy.average(toy_hists, axis=0)
    toy_stdevs = numpy.std(toy_hists, axis=0)
    #print(' '.join([f'{v:.02f}' for v in toy_averages]))
    #print(' '.join([f'{v:.02f}' for v in toy_stdevs]))
    toy_llvs = -numpy.log(scipy.stats.norm.pdf(toy_hists, loc=toy_averages, scale=toy_stdevs)).sum(axis=1)
    llv_avg = numpy.average(toy_llvs)
    llv_std = numpy.std(toy_llvs)

    vals_llv = -numpy.log(scipy.stats.norm.pdf(vals, loc=toy_averages, scale=toy_stdevs)).sum()
    vals_llv_zval = (vals_llv - llv_avg) / llv_std
    if vals_llv_zval > 2:
        print(f'Predicted Bias of {predicted_bias} excluded at {vals_llv_zval:.02f} sigma')
    else:
        print(f'Predicted Bias of {predicted_bias} compatible at {vals_llv_zval:.02f} sigma')


    fig, ax = plt.subplots()
    ax.hist(toy_llvs)
    ax.axvline(x=vals_llv, color='purple')
    ax.axvline(x=llv_avg, color='blue', ls='--')
    ax.axvline(x=llv_avg+1*llv_std, color='cyan', ls='--')
    ax.axvline(x=llv_avg+2*llv_std, color='green', ls='--')
    ax.axvline(x=llv_avg+3*llv_std, color='yellow', ls='--')
    ax.axvline(x=llv_avg+4*llv_std, color='red', ls='--')
    ax.axvline(x=llv_avg+5*llv_std, color='maroon', ls='--')

    #ax.grid()
    #ax.set_ylim(yrange)

    plt.savefig(name_prefix+'_LogLikeLihood.pdf')
    return expec_vals, toy_stdevs


def draw_sig_bgd(vals=None, err=None, expec_vals=None, expec_err=None, yrange=None, bins=None, name_prefix=''):

    # Analytic Confidence Interval Check
    #p = expec_vals[0]/total_events
    #q = expec_vals[1]/total_events
    #N = total_events
    #xvals = list(range(total_events+1))
    #Pvals = []
    #for n in xvals:
    #    P = math.comb(N,n) * p**n * q**(N-n)
    #    Pvals.append(P)
    #Pvals = numpy.array(Pvals)
    #expec_avg = int(numpy.average(xvals, weights=Pvals))
    #print(expec_vals)
    #print(expec_avg)

    #Psums = []
    #for i in range(total_events+1):
    #    mini = expec_avg-i
    #    if mini < 0: mini = 0
    #    maxi = expec_avg+i
    #    if total_events < maxi: maxi = total_events
    #    Psum = Pvals[mini:maxi+1].sum()
    #    Psums.append(Psum)
    #    if Psum >= 1: break

    #for conf95_Dx, p in enumerate(Psums):
    #    if p > 0.95: break
    #conf95_lo = expec_avg - conf95_Dx
    #conf95_hi = expec_avg + conf95_Dx
    #print(conf95_Dx)
    #if vals[0] < conf95_lo or conf95_hi < vals[0]:
    #    print('Incompatible')
    #else:
    #    print('Compatible')

    fig, axes = plt.subplots(2, gridspec_kw={'height_ratios':(4,1)}, sharex=True)

    axes[0].errorbar(bins[:-1], vals,  marker='.', ls='-', color='purple', capsize=2, drawstyle='steps-mid', label='vals')
    axes[0].errorbar(bins[:-1], expec_vals, yerr=expec_err, marker=',', ls='-', color='red', capsize=2, elinewidth=.5, drawstyle='steps-mid', label='expectation')
    #axes[0].errorbar(bins[:-1], expec_sig_vals, marker=',', ls='--', color='blue')
    #axes[0].errorbar(bins[:-1], expec_bgd_vals, yerr=expec_bgd_err, marker=',', ls='--', color='green')
    #axes[0].hlines(conf95_lo, bins[0], bins[-2], linestyle='-', color='black')
    #axes[0].hlines(conf95_hi, bins[0], bins[-2], linestyle='-', color='black')

    axes[0].grid()
    axes[0].set_ylim(yrange)
    axes[0].legend()

    #rat.hlines(0, bins[0], bins[-1], linestyle='-', color='black')
    #rat.hlines(2, bins[0], bins[-1], linestyle='dotted', color='green')
    #rat.errorbar( bins[:-1], (expec_sig_vals-vals)/err, marker=',', ls='--', color='red')
    #rat.grid()
    #rat.set_xlim(bins[0],bins[-1])
    #rat.set_ylim(-3,3)
    #rat.set_yticks([-3,-2,-1,0,1,2,3])

    #rat.set_xticks(bins[:-1])
    #rat.set_xticks(rat.get_xticks()+0.5, minor=True)
    #rat.set_xticklabels('') # Clear major tick labels
    #rat.set_xticklabels(rat.get_xticks(), minor=True, ha='center', fontsize=12)
    #axes[0].set_title(f'Biased Face = {_biased_num}')

    plt.savefig(name_prefix+'_hist_rolls.pdf')




def roll(_biased_num, bias_amount):
    if _biased_num is None:
        roll = random.randint(0,_die_faces-1)
    else:
        val = random.uniform(0,_die_faces+bias_amount-1)
        roll = int(val) if val < _die_faces else _biased_num
        
    return roll


def main():
    num_bgd_dice = 0
    bias_amount = 2
    predicted_bias = 2
    num_rolls = 600
    var_edges = range(0,_die_faces+1)
    name_prefix = 'dice'

    signal_rolls = [ roll(_biased_num, bias_amount) for i in range(num_rolls) ]
    bgd_rolls = [ roll(None,None) for i in range(num_rolls*num_bgd_dice) ]
    all_rolls = numpy.array(signal_rolls + bgd_rolls)

    vals, _ = numpy.histogram(all_rolls, bins=var_edges)
    err = numpy.sqrt(vals)

    signal_total_P = _die_faces+predicted_bias-1
    expec_sig_vals = numpy.array([ 1 if i != _biased_num else predicted_bias for i in range(_die_faces) ])
    expec_sig_vals = expec_sig_vals/signal_total_P * num_rolls
    bgd_estimate, _ = numpy.histogram(var_edges[:-1], weights=[num_bgd_dice*num_rolls/_die_faces]*_die_faces, bins=var_edges)
    expec_vals = expec_sig_vals+bgd_estimate

    max_likelihood_bias = evaluate_max_likelihood(vals=vals, bgd_est=bgd_estimate, bins=var_edges, name_prefix=name_prefix)
    expec_err = draw_max_likelihood(vals=vals, expec_vals=expec_vals, predicted_bias=predicted_bias, bins=var_edges, name_prefix=name_prefix)
    draw_sig_bgd(vals=vals, err=err, expec_vals=expec_vals, expec_err=expec_err, bins=var_edges, name_prefix=name_prefix)

            


main()
