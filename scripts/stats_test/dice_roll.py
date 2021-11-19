#!/usr/bin/python
import sys
import random
import math
import statistics
import numpy
import scipy.stats
from matplotlib import pyplot as plt

_die_faces = 6


def draw_sig_bgd(vals=None, err=None, expec_vals=None, expec_err=None, yrange=None, bins=None, name_prefix=''):
    total_events = vals.sum()

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


    num_toys = 1000
    toy_hists = []
    for i in range(num_toys):
        toy_distro = numpy.random.choice(bins[:-1], size=total_events, p=expec_vals/expec_vals.sum())
        hist, _ = numpy.histogram(toy_distro, bins=bins)
        toy_hists.append(hist)
    toy_hists = numpy.array(toy_hists)

    toy_averages = numpy.average(toy_hists, axis=0)
    toy_stdevs = numpy.std(toy_hists, axis=0)
    toy_llvs = numpy.log(scipy.stats.norm.pdf(toy_hists, loc=toy_averages, scale=toy_stdevs)).sum(axis=1)
    vals_llv = numpy.log(scipy.stats.norm.pdf(vals, loc=toy_averages, scale=toy_stdevs)).sum()
    pvalue = (toy_llvs < vals_llv).sum() / num_toys
    print(pvalue, pvalue < .05)
    exit()

        

    fig, axes = plt.subplots(2, gridspec_kw={'height_ratios':(4,1)}, sharex=True)

    axes[0].errorbar(bins[:-1], vals,  marker='.', ls='-', color='purple', capsize=2, drawstyle='steps-mid', label='vals')
    axes[0].errorbar(bins[:-1], expec_vals, yerr=toy_stdev, marker=',', ls='-', color='red', capsize=2, elinewidth=.5, drawstyle='steps-mid', label='expectation')
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
    #axes[0].set_title(f'Biased Face = {biased_num}')

    plt.savefig(name_prefix+'_hist_rolls.pdf')


def draw_max_likelihood(vals=None, err=None, bgd=None, yrange=None, bins=None, name_prefix=''):
    mu = numpy.linspace(-100,100,100)
    #logSeries = - numpy.log(numpy.outer(mu,vals)+bgd)
    logSeries = (numpy.outer(mu,vals)+bgd)
    logSum = logSeries.sum(axis=1)

    fig, ax = plt.subplots()
    ax.errorbar(mu, logSum, marker='.', ls='--', color='purple')

    #ax.grid()
    #ax.set_ylim(yrange)

    plt.savefig(name_prefix+'_LogLikeLihood.pdf')


def roll(biased_num, bias_amount):
    if biased_num is None:
        roll = random.randint(1,_die_faces)
    else:
        val = random.uniform(1,_die_faces+bias_amount)
        roll = int(val) if val <= _die_faces+1 else biased_num
        
    return roll


def main():
    num_bgd_dice = 0
    biased_num = 1
    bias_amount = 2
    predicted_bias = 1
    num_rolls = 60
    var_edges = range(1,_die_faces+2)
    name_prefix = 'dice'


    signal_rolls = [ roll(biased_num, bias_amount) for i in range(num_rolls) ]
    bgd_rolls = [ roll(None,None) for i in range(num_rolls*num_bgd_dice) ]
    all_rolls = numpy.array(signal_rolls + bgd_rolls)


    vals, _ = numpy.histogram(all_rolls, bins=var_edges)
    err = numpy.sqrt(vals)


    signal_total_P = _die_faces+predicted_bias-1
    expec_sig_vals = numpy.array([ 1 if i+1 != biased_num else predicted_bias for i in range(_die_faces) ])
    expec_sig_vals = expec_sig_vals/signal_total_P * num_rolls

    expec_bgd_vals = numpy.full(_die_faces, 1/_die_faces) * len(bgd_rolls)
    expec_bgd_err  = numpy.full(_die_faces, math.sqrt(num_bgd_dice*num_rolls) )

    expec_vals = expec_sig_vals#+expec_bgd_vals
    expec_err = expec_bgd_err

    draw_sig_bgd(vals=vals, err=err, expec_vals=expec_vals, expec_err=expec_err, bins=var_edges, name_prefix=name_prefix)
    #draw_max_likelihood(vals=vals, bgd=expec_bgd_vals, bins=var_edges, name_prefix=name_prefix)


def alt():
    p = 0.5
    q = 1 - p
    N = 6
    L = 0
    for l in range(int(N/2),-1,-1):
        m = N - l
        Pt = 0
        n1 = 0
        for n in range(0,l+1):
            P = math.comb(N,n) * p**n * q**(N-n)
            Pt += P
            n1 += 1
        n2 = 0
        for n in range(m,N+1):
            P = math.comb(N,n) * p**n * q**(N-n)
            Pt += P
            n2 += 1
        print(f' 0-{l:2}:{n1:2}, {m:2}-{N:2}:{n2:2} / {N}: {Pt:.04f}')
            





    

main()
#alt()
