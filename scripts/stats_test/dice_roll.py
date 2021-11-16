#!/usr/bin/python
import sys
import random
import math
import statistics
import numpy
from matplotlib import pyplot as plt

_die_faces = 6

def roll(biased_num, bias_amount):
    if biased_num is None:
        roll = random.randint(1,_die_faces)
    else:
        val = random.uniform(1,_die_faces+bias_amount)
        roll = int(val) if val <= _die_faces+1 else biased_num
        
    return roll


def main():
    num_bgd_dice = 0
    biased_num = 4
    bias_amount = 2
    predicted_bias = 1.5
    num_rolls = 20



    signal_rolls = [ roll(biased_num, bias_amount) for i in range(num_rolls) ]
    bgd_rolls = [ roll(None,None) for i in range(num_rolls*num_bgd_dice) ]
    all_rolls = numpy.array(signal_rolls + bgd_rolls)

    vals, bins = numpy.histogram(all_rolls, bins=range(1,_die_faces+2))
    err = numpy.sqrt(vals)


    signal_total_P = _die_faces+predicted_bias-1
    expec_sig_vals = numpy.array([ 1 if i+1 != biased_num else predicted_bias for i in range(_die_faces) ])
    expec_sig_vals = expec_sig_vals/signal_total_P * num_rolls

    expec_bgd_vals = numpy.full(6, 1/_die_faces) * num_bgd_dice*num_rolls
    expec_bgd_err  = numpy.full(6, math.sqrt(num_bgd_dice*num_rolls) )

    expec_vals = expec_sig_vals+expec_bgd_vals
    expec_err = expec_bgd_err


    fig, axes = plt.subplots(2, gridspec_kw={'height_ratios':(4,1)}, sharex=True)

    axes[0].errorbar(bins[:-1]+0.5, vals, yerr=err, marker='.', ls='--', color='purple', capsize=2)
    axes[0].errorbar(bins[:-1]+0.5, expec_vals, yerr=expec_err, marker=',', ls='--', color='red', capsize=2, elinewidth=.5)
    #axes[0].errorbar(bins[:-1]+0.5, expec_sig_vals, marker=',', ls='--', color='blue')
    #axes[0].errorbar(bins[:-1]+0.5, expec_bgd_vals, yerr=expec_bgd_err, marker=',', ls='--', color='green')

    axes[0].grid()
    axes[0].set_ylim(0,len(all_rolls)/2)

    #rat.hlines(0, bins[0], bins[-1], linestyle='-', color='black')
    #rat.hlines(2, bins[0], bins[-1], linestyle='dotted', color='green')
    #rat.errorbar( bins[:-1]+0.5, (expec_sig_vals-vals)/err, marker=',', ls='--', color='red')
    #rat.grid()
    #rat.set_xlim(bins[0],bins[-1])
    #rat.set_ylim(-3,3)
    #rat.set_yticks([-3,-2,-1,0,1,2,3])

    #rat.set_xticks(bins[:-1])
    #rat.set_xticks(rat.get_xticks()+0.5, minor=True)
    #rat.set_xticklabels('') # Clear major tick labels
    #rat.set_xticklabels(rat.get_xticks(), minor=True, ha='center', fontsize=12)
    #axes[0].set_title(f'Biased Face = {biased_num}')

    plt.savefig('hist_rolls.pdf')
    

main()
