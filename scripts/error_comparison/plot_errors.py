#!/usr/bin/python

import os
import re
import json
import numpy
import matplotlib
from matplotlib import pyplot as plt

sys_regions = ['East', 'North', 'South', 'West']
    
for f in sorted(os.listdir('workspaces')):
    if '.json' not in f: continue

    k = re.search(r".*kl_([^_]*)_k2v_([^_]*)_k1v_([^_]*)\.json", f).groups()
    coupling_name = f'kl{float(k[0]):.1f}_kvv{float(k[1]):.1f}_kv{float(k[2]):.1f}'.replace('.','p')
    coupling_title = r'$\kappa_{\lambda},\kappa_{2V},\kappa_{V}$ = 'f'{float(k[0]):.1f}, {float(k[1]):.1f}, {float(k[2]):.1f}'
    json_file = json.load(open('workspaces/'+f))
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
    for category_index, observed in enumerate(json_file['observations']):
        full_stats[category_index]['data']['Observed'] = numpy.array(observed['data'])
        
    for category_index, hypothesis in enumerate(json_file['channels']):
        data = full_stats[category_index]['data']

        vbf_signal = hypothesis['samples'][0]
        data['Signal'] = numpy.array(vbf_signal['data'])
        data['Signal Stat Error'] = numpy.array(vbf_signal['modifiers'][0]['data'])

        bgd_prediction = hypothesis['samples'][1]
        data['Background'] = numpy.array(bgd_prediction['data'])
        data['Background Stat Error'] = numpy.array(bgd_prediction['modifiers'][4]['data'])
        for shape_index, shape in enumerate(sys_regions):
            shape_systematic_hi = numpy.array(bgd_prediction['modifiers'][shape_index]['data']['hi_data'])
            data['Background Sys '+shape] = abs(shape_systematic_hi - full_stats[category_index]['data']['Observed'])

    for cat_index, category in enumerate(full_stats):
        fig, (ax, rat, errplt) = plt.subplots(3, gridspec_kw={'height_ratios':(1,1,1)}, sharex=True)
        bins = category['bins']
        data = category['data']
        ax.plot(bins, data['Observed'], label='Observed', color='black')
        ax.errorbar(bins, data['Signal'], 
            yerr=data['Signal Stat Error'], label='Signal', color='green')
        ax.errorbar(bins, data['Background'],
            yerr=data['Background Stat Error'], label='Background', color='blue')
        ax.legend()
        ax.set_ylabel('Event Count')

        safe_obs = data['Observed'].copy()
        safe_obs[safe_obs < 1] = float('inf')
        rat.plot(bins, data['Signal Stat Error']/data['Signal'], color='green', label='Sig Stat Error')
        rat.plot(bins, data['Background Stat Error']/data['Background'], color='blue', label='Bgd Stat Error')
        for shape_index, shape in enumerate(sys_regions):
            key = 'Background Sys '+shape
            rat.plot(bins, data[key]/safe_obs, label='Sys '+shape)
        rat.set_ylim(0,1)
        rat.legend(fontsize=5)
        rat.set_ylabel('Error/Prediction')

        errplt.plot(bins, data['Signal Stat Error'], color='green', label='Sig Stat Error')
        errplt.plot(bins, data['Background Stat Error'], color='blue', label='Bgd Stat Error')
        for shape_index, shape in enumerate(sys_regions):
            key = 'Background Sys '+shape
            errplt.plot(bins, data[key], label='Sys '+shape)
        errplt.legend(fontsize=5)
        errplt.set_ylabel('Error Value')

        errplt.set_xlabel('$M_{HH} (GeV)$')

        title = coupling_title + '\n' + category['title']
        fig.suptitle(title)
        plt.savefig('out/error_comparison_'+coupling_name+f'cat{cat_index}.pdf')
        plt.close()

    ##x = 
    #for key, data in workspace_data.items():
    #    ax.plot(data,label=key)
    ##rat.plot(data/data_average)
    ##ax.set_ylabel('data')
    ##rat.set_ylabel('data / avg(data)')
    ##for d,l in zip(datasets,labels):
    ##    ax.plot(d, label=l)

    #title = coupling_title
    #fig.suptitle(title)
    #ax.legend()
    #plt.savefig('error_comparison_'+coupling_name+'.pdf')
    #plt.close()
