#!/usr/bin/python

import os
import re
import json
import numpy
import matplotlib
import sympy
from matplotlib import pyplot as plt

_k2v = sympy.Symbol('KVV')
_kl = sympy.Symbol('KL')
_kv = sympy.Symbol('KV')

_full_scan_terms = [
    lambda kl,k2v,kv: kv**2 * kl**2,
    lambda kl,k2v,kv: kv**4,
    lambda kl,k2v,kv: k2v**2,
    lambda kl,k2v,kv: kv**3 * kl,
    lambda kl,k2v,kv: k2v * kl * kv,
    lambda kl,k2v,kv: kv**2 * k2v
]

_kl_scan_terms = [
    lambda kl,k2v,kv: kl**2,
    lambda kl,k2v,kv: kl,
    lambda kl,k2v,kv: 1
]


def construct_coupling_info(kappas):
    coupling_name = f'kl{kappas[0]:.1f}_kvv{kappas[1]:.1f}_kv{kappas[2]:.1f}'.replace('.','p')
    coupling_title = r'$\kappa_{\lambda},\kappa_{2V},\kappa_{V}$ = 'f'{float(kappas[0]):.1f}, {float(kappas[1]):.1f}, {float(kappas[2]):.1f}'
    return coupling_name, coupling_title


def get_coupling_info(name):
    kgroups = re.search(r".*kl_([^_]*)_k2v_([^_]*)_k1v_([^_]*)", name).groups()
    kappas = tuple([ float(k) for k in kgroups ])
    return kappas, *construct_coupling_info(kappas)



def get_amplitude_function( basis_parameters, base_equations):
    basis_states = [ [ sympy.Rational(param) for param in basis ] for basis in basis_parameters ]
    combination_matrix = sympy.Matrix([ [ g(*base) for g in base_equations ] for base in basis_states])
    if combination_matrix.det() == 0: return None

    inversion = combination_matrix.inv()
    term_vector = sympy.Matrix([ [g(_k2v,_kl,_kv)] for g in base_equations ])
    amplitudes = sympy.Matrix([ sympy.Symbol(f'A{n}') for n in range(len(base_equations)) ])
    final_weight = term_vector.T * inversion
    reweight_vector_function = sympy.lambdify([_k2v, _kl, _kv], final_weight, 'numpy')
    return reweight_vector_function


def generate_combination(base_vals, coupling, ggF=False):
    base_equations = _kl_scan_terms if ggF else _full_scan_terms
    reweight_vector = get_amplitude_function( list(base_vals), base_equations )(*coupling)[0]
    coupling_name, coupling_title = construct_coupling_info(coupling)
    signal = {'name':coupling_name, 'title':coupling_title}

    base_weights = [v['values'] for v in base_vals.values()]
    signal['values'] = numpy.array([ w*m for w,m in zip(base_weights, reweight_vector) ]).sum(axis=0)

    base_staterrs = [v['stat_err'] for v in base_vals.values()]
    signal['stat_err'] = numpy.sqrt(numpy.array([ (w*m)**2 for w,m in zip(base_staterrs, reweight_vector) ]).sum(axis=0))

    base_systerrs = [v['syst_err'] for v in base_vals.values()]
    signal['syst_err'] = numpy.sqrt(numpy.array([ (w*m)**2 for w,m in zip(base_systerrs, reweight_vector) ]).sum(axis=0))

    return signal



def main(): 
    #numpy.set_printoptions(precision=6, linewidth=800, threshold=100, sign=' ')

    full_stats = [
        {'title': '$\eta < 1.5$',
            'bins': [400, 450, 495, 545, 600, 660, 725, 800, 880],#, 965],
            'data': {}
        },
        {'title': '$\eta \geq 1.5$',
            'bins': [400, 445, 485, 530, 580, 630, 685, 750, 815, 890, 970, 1055, 1150, 1255, 1370],#, 1490],
            'data': {}
        }
    ]

    multi_channel_workspace = 'workspaces/wksp_VBF_chan_allNP_samps_vbf_pd_vbf_inc161718.json'
    json_file = json.load(open(multi_channel_workspace))
    
    for category_index, observed in enumerate(json_file['observations']):
        full_stats[category_index]['data']['Observed'] = numpy.array(observed['data'])
        
    for category_index, hypothesis in enumerate(json_file['channels']):
        values = full_stats[category_index]['data']

        # Retrieve VBF Signal Hypothesis
        vbf_variations = {}
        for vbf_index in range(6):
            vbf_signal = hypothesis['samples'][vbf_index]
            kappas, coupling_name, coupling_title = get_coupling_info(vbf_signal['name'])
            vbf_variations[kappas] = {'name':coupling_name, 'title':coupling_title}
            vbf_variations[kappas]['values'] = numpy.array(vbf_signal['data'])

            total_sys = numpy.zeros_like(vbf_variations[kappas]['values'])
            for error in vbf_signal['modifiers']:
                if error['data'] is None: continue
                if error['type'] == 'shapesys':
                    vbf_variations[kappas]['stat_err'] = numpy.array(error['data'])
                elif error['type'] == 'normsys':
                    hi, lo = error['data'].values()
                    adjust_hi = hi*vbf_variations[kappas]['values']
                    adjust_lo = lo*vbf_variations[kappas]['values']
                    norm_err = abs(adjust_hi-adjust_lo)
                    #print('N: '+' '.join([f'{v:5.4f}' for v in norm_err]))
                    total_sys += norm_err**2
                else:
                    hi, lo = error['data'].values()
                    hi_err = abs( numpy.array(hi) - vbf_variations[kappas]['values'] )
                    lo_err = abs( numpy.array(lo) - vbf_variations[kappas]['values'] )
                    err_vals = (hi_err+lo_err)/2
                    #err_vals = abs(numpy.array(hi)-numpy.array(lo))
                    #print('H: '+' '.join([f'{v:5.4f}' for v in err_vals]))
                    total_sys += err_vals**2
            vbf_variations[kappas]['syst_err'] = numpy.sqrt(total_sys)
        #for k,v in vbf_variations.items():
        #    print(k)
        #    print('V: '+' '.join([f'{i:5.4f}' for i in v['values']]))
        #    print('E: '+' '.join([f'{i:5.4f}' for i in v['stat_err']]))
        #    print('S: '+' '.join([f'{i:5.4f}' for i in v['syst_err']]))
        #    print()
        values['vbf_variations'] = vbf_variations


        # Retrieve Background Model
        bgd_prediction = hypothesis['samples'][6]
        values['background'] = {}
        values['background']['values'] = numpy.array(bgd_prediction['data'])
        values['background']['stat_err'] = numpy.array(bgd_prediction['modifiers'][4]['data'])

        total_sys = numpy.zeros_like(values['background']['values'])
        for syst in bgd_prediction['modifiers'][:-1]:
            hi, lo = syst['data'].values()
            hi_err = abs( numpy.array(hi) - values['background']['values'] )
            lo_err = abs( numpy.array(lo) - values['background']['values'] )
            err_vals = (hi_err+lo_err)/2
            #err_vals = abs(numpy.array(hi)-numpy.array(lo))
            total_sys += err_vals**2
        values['background']['syst_err'] = numpy.sqrt(total_sys)
        #print('V: '+' '.join([f'{i:5.2f}' for i in values['background']['values']]))
        #print('E: '+' '.join([f'{i:5.2f}' for i in values['background']['stat_err']]))
        #print('S: '+' '.join([f'{i:5.2f}' for i in values['background']['syst_err']]))


        # Retrieve ggF Background Hypothesis
        ggf_variations = {}
        for ggf_index in range(7,10):
            ggf_signal = hypothesis['samples'][ggf_index]
            kappas, coupling_name, coupling_title = get_coupling_info(ggf_signal['name'])
            ggf_variations[kappas] = {'name':coupling_name, 'title':coupling_title}
            ggf_variations[kappas]['values'] = numpy.array(ggf_signal['data'])

            total_sys = numpy.zeros_like(ggf_variations[kappas]['values'])
            for error in ggf_signal['modifiers']:
                if error['data'] is None: continue
                if error['type'] == 'shapesys':
                    ggf_variations[kappas]['stat_err'] = numpy.array(error['data'])
                elif error['type'] == 'normsys':
                    hi, lo = error['data'].values()
                    adjust_hi = hi*ggf_variations[kappas]['values']
                    adjust_lo = lo*ggf_variations[kappas]['values']
                    norm_err = abs(adjust_hi-adjust_lo)
                    #print('N: '+' '.join([f'{v:5.4f}' for v in norm_err]))
                    #total_sys += norm_err**2
                else:
                    hi, lo = error['data'].values()
                    err_vals = abs(numpy.array(hi)-numpy.array(lo))
                    #print('H: '+' '.join([f'{v:5.4f}' for v in err_vals]))
                    #total_sys += err_vals**2
            ggf_variations[kappas]['syst_err'] = numpy.sqrt(total_sys)
        #for k,v in ggf_variations.items():
        #    print(k)
        #    print('V: '+' '.join([f'{v:5.4f}' for v in v['values']]))
        #    print('E: '+' '.join([f'{v:5.4f}' for v in v['stat_err']]))
        #    print('S: '+' '.join([f'{v:5.4f}' for v in v['syst_err']]))
        #    print()
        values['ggf_variations'] = ggf_variations




    variation_list = [(1,1,1), (10,1,1), (1,3,1)]
    for variation in variation_list:
        for cat_index, category in enumerate(full_stats):

            fig, (ax, errplt, rat) = plt.subplots(3, gridspec_kw={'height_ratios':(1,1,1)}, sharex=True)
            bins = category['bins']
            values = category['data']
            sig = generate_combination(values['vbf_variations'], variation)
            ggf = generate_combination(values['ggf_variations'], variation, ggF=True)
            #sig = values['vbf_variations'][variation]
            #ggf = values['ggf_variations'][(1,1,1)]
            bgd = values['background']

            ax.plot(bins, values['Observed'], label='Observed', color='black')
            ax.errorbar(bins, sig['values'], 
                yerr=sig['stat_err'], label='Sig', color='red')
            ax.errorbar(bins, ggf['values'], 
                yerr=ggf['stat_err'], label='ggF', color='green')
            ax.errorbar(bins, bgd['values'],
                yerr=bgd['stat_err'], label='Bgd', color='blue')
            ax.legend(fontsize=10, loc='upper right', ncol=4, bbox_to_anchor=(1,1.3))
            ax.set_ylim(1e-3,1e2)
            ax.set_yscale('log')
            ax.set_ylabel('Event Count')

            errplt.plot(bins, sig['stat_err'], color='red', label='Sig Stat Error')
            errplt.plot(bins, sig['syst_err'], color='orange', label='Sig Syst Error')
            errplt.plot(bins, ggf['stat_err'], color='green', label='ggF Stat Error')
            errplt.plot(bins, ggf['syst_err'], color='lime', label='ggF Syst Error')
            errplt.plot(bins, bgd['stat_err'], color='blue', label='Bgd Stat Error')
            errplt.plot(bins, bgd['syst_err'], color='cyan', label='Bgd Syst Error')
            errplt.legend(fontsize=8, loc='upper center', ncol=3)
            errplt.set_ylabel('Yield Error')

            #safe_obs = values['Observed'].copy()
            #safe_obs[safe_obs < 1] = float('inf')
            #safe = lambda d: numpy.array([ float('inf') if v < 1e-12 else v for v in d ])
            rat.plot(bins, sig['stat_err']/sig['values'], color='red', label='Sig Stat Error')
            rat.plot(bins, sig['syst_err']/sig['values'], color='orange', label='Sig Syst Error')
            rat.plot(bins, ggf['stat_err']/ggf['values'], color='green', label='ggF Stat Error')
            rat.plot(bins, ggf['syst_err']/ggf['values'], color='lime', label='ggF Syst Error')
            rat.plot(bins, bgd['stat_err']/bgd['values'], color='blue', label='Bgd Stat Error')
            rat.plot(bins, bgd['syst_err']/bgd['values'], color='cyan', label='Bgd Syst Error')

            rat.set_ylim(0,2)
            rat.set_ylabel('Relative Error')
            #rat.legend(fontsize=9, loc='upper center', ncol=3)

            rat.set_xlabel('$M_{HH} (GeV)$')

            title = sig['title'] + ' $\;\emdash\;$ ' + category['title']
            fig.suptitle(title)
            plt.savefig('out/error_comparison_'+sig['name']+f'cat{cat_index}.pdf')
            plt.close()

main()
