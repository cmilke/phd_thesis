#!/usr/bin/python

import os
import re
import json
import numpy
import matplotlib
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

workspace_data = {}
labels = []
def embed(d,s,e):
    if s in d: d[s].append(e)
    else: d[s] = [e]
    
for f in sorted(os.listdir('.')):
    if '.json' not in f: continue
    k = re.search(r".*kl_([^_]*)_k2v_([^_]*)_k1v_([^_]*)\.json", f).groups()
    #label = r'$\kappa_{\lambda},\kappa_{2V},\kappa_{V}$ = 'f'{float(k[0]):.0f}, {float(k[1]):.2f}, {float(k[2]):.0f}'
    label = r'$\kappa_{2V}$ = 'f'{float(k[1]):.2f}'
    labels.append(label)
    j = json.load(open(f))
    for channel in j['channels']:
        for sample in channel['samples']:
            plot_name = channel['name'] + '\n '+sample['name'] + ': data'
            embed(workspace_data, plot_name, sample['data'])
            #if 'modifiers' not in sample: continue
            for mod in sample['modifiers']:
                if mod['data'] is None: continue
                if type(mod['data']) == list:
                    plot_name = channel['name'] + '\n '+sample['name']+': '+mod['name']
                    embed(workspace_data, plot_name, mod['data'])
                else:
                    for hilo, data in mod['data'].items():
                        plot_name = channel['name'] + '\n '+sample['name']+': '+mod['name']+' - '+hilo
                        embed(workspace_data, plot_name, data)

with PdfPages('validation_vbf.pdf') as output:
#with PdfPages('validation.pdf') as output:
    for title, datasets in workspace_data.items():
        if 'vbfSig' not in title: continue
        print(title)

        fig, (ax, rat) = plt.subplots(2, gridspec_kw={'height_ratios':(2,1)}, sharex=True)
        data = numpy.array(datasets).transpose()
        data_average = numpy.average(data,axis=1)[:,None]
        ax.plot(data,label=labels)
        rat.plot(data/data_average)
        ax.set_ylabel('data')
        rat.set_ylabel('data / avg(data)')
        #for d,l in zip(datasets,labels):
        #    ax.plot(d, label=l)

        fig.suptitle(title)
        ax.legend()
        output.savefig()
        plt.close()
