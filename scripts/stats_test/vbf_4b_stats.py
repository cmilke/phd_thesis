#!/usr/bin/python
import sys
import random
import statistics
import pickle
import numpy
import uproot
from matplotlib import pyplot as plt

def load_data(var_edges):
    data_files = [
        '/home/cmilke/Documents/dihiggs/background/output_dir_20-20-20_bst100_VBF_DS/data16_Xhh_45_NN_100_bootstraps.root',
        '/home/cmilke/Documents/dihiggs/background/output_dir_20-20-20_bst100_VBF_DS/data17_Xhh_45_NN_100_bootstraps.root',
        '/home/cmilke/Documents/dihiggs/background/output_dir_20-20-20_bst100_VBF_DS/data18_Xhh_45_NN_100_bootstraps.root',
    ]
    key='m_hh'
    events = []
    for f in data_files:
        tree_name = 'sig'
        rootfile = uproot.open(f)
        ttree = rootfile[tree_name]

        pass_vbf_sel = ttree['pass_vbf_sel'].array()
        x_wt_tag = ttree['X_wt_tag'].array() > 1.5
        ntag = ttree['ntag'].array() == 3 # 3b1f region
        valid_event = numpy.logical_and.reduce( (pass_vbf_sel, x_wt_tag, ntag) )

        kinvals = ttree['m_hh'].array()[valid_event]
        weights = ttree['mc_sf'].array()[valid_event]
        lumi_weights = weights

        new_events = numpy.array([kinvals,lumi_weights])
        events.append(new_events)

    event_var, event_weights = numpy.concatenate(events, axis=1)

    weights = numpy.histogram(event_var, bins=var_edges, weights=event_weights)[0]
    errors = numpy.sqrt(numpy.histogram(event_var, bins=var_edges, weights=event_weights**2)[0])
    return (weights, errors)


def load_bgd(var_edges):
    data_files = [
        (16,'/home/cmilke/Documents/dihiggs/background/output_dir_20-20-20_bst100_VBF_DS/data16_Xhh_45_NN_100_bootstraps.root'),
        (17,'/home/cmilke/Documents/dihiggs/background/output_dir_20-20-20_bst100_VBF_DS/data17_Xhh_45_NN_100_bootstraps.root'),
        (18,'/home/cmilke/Documents/dihiggs/background/output_dir_20-20-20_bst100_VBF_DS/data18_Xhh_45_NN_100_bootstraps.root')
    ]
    key='m_hh'
    events = []
    for y,f in data_files:
        tree_name = 'sig'
        rootfile = uproot.open(f)
        med_norm = rootfile[f'NN_norm_3b1f_bstrap_med_{y}'].value
        VR_med_norm = rootfile[f'NN_norm_VRderiv_3b1f_bstrap_med_{y}'].value
        ttree = rootfile[tree_name]

        pass_vbf_sel = ttree['pass_vbf_sel'].array()
        rw_pass = ttree['rw_to_3b1f'].array()
        x_wt_tag = ttree['X_wt_tag'].array() > 1.5
        ntag = ttree['ntag'].array() == 2
        valid_event = numpy.logical_and.reduce( (pass_vbf_sel, rw_pass, x_wt_tag, ntag) )

        kinvals = ttree['m_hh'].array()[valid_event]
        NN_weights = ttree[f'NN_d231f_weight_bstrap_med_{y}'].array()[valid_event]*med_norm
        VR_weights = ttree[f'NN_d231f_weight_VRderiv_bstrap_med_{y}'].array()[valid_event]*med_norm

        new_events = numpy.array([kinvals,NN_weights,VR_weights])
        events.append(new_events)

    event_var, event_weights, event_VR_weights = numpy.concatenate(events, axis=1)
    
    weights = numpy.histogram(event_var, bins=var_edges, weights=event_weights)[0]
    alt_weights = numpy.histogram(event_var, bins=var_edges, weights=event_VR_weights)[0]
    sys_errors = abs(alt_weights - weights)
    stat_errors = numpy.sqrt(numpy.histogram(event_var, bins=var_edges, weights=event_weights**2)[0])
    errors = numpy.sqrt( sys_errors**2 + stat_errors**2)
    return (weights, errors)


def load_signal(var_edges):
    #signal_files = [
    #    '/home/cmilke/Documents/dihiggs/nano_ntuples/2021_aug_test/VBF_nonres2021_502970_MC16a-2015-2016_NanoNTuple.root',
    #    '/home/cmilke/Documents/dihiggs/nano_ntuples/2021_aug_test/VBF_nonres2021_502970_MC16d-2017_NanoNTuple.root',
    #    '/home/cmilke/Documents/dihiggs/nano_ntuples/2021_aug_test/VBF_nonres2021_502970_MC16e-2018_NanoNTuple.root'
    #]
    signal_files = [
        '/home/cmilke/Documents/dihiggs/nano_ntuples/2021_aug_test/VBF_nonres2021_502975_MC16a-2015-2016_NanoNTuple.root',
        '/home/cmilke/Documents/dihiggs/nano_ntuples/2021_aug_test/VBF_nonres2021_502975_MC16d-2017_NanoNTuple.root',
        '/home/cmilke/Documents/dihiggs/nano_ntuples/2021_aug_test/VBF_nonres2021_502975_MC16e-2018_NanoNTuple.root'
    ]

    key='m_hh'
    events = []
    for f in signal_files:
        tree_name = 'sig'
        rootfile = uproot.open(f)
        ttree = rootfile[tree_name]

        pass_vbf_sel = ttree['pass_vbf_sel'].array()
        x_wt_tag = ttree['X_wt_tag'].array() > 1.5
        ntag = ttree['ntag'].array() >= 4
        valid_event = numpy.logical_and.reduce( (pass_vbf_sel, x_wt_tag, ntag) )

        kinvals = ttree['m_hh'].array()[valid_event]
        weights = ttree['mc_sf'].array()[:,0][valid_event]
        run_number = ttree['run_number'].array()[valid_event]

        mc2015 = ( run_number < 296939 ) * 3.2
        mc2016 = ( numpy.logical_and(296939 < run_number, run_number < 320000) ) * 24.6
        mc2017 = ( numpy.logical_and(320000 < run_number, run_number < 350000) ) * 43.65
        mc2018 = ( numpy.logical_and(350000 < run_number, run_number < 370000) ) * 58.45
        all_years = mc2015 + mc2016 + mc2017 + mc2018
        lumi_weights = weights * all_years

        new_events = numpy.array([kinvals,lumi_weights])
        events.append(new_events)

    event_var, event_weights = numpy.concatenate(events, axis=1)

    weights = numpy.histogram(event_var, bins=var_edges, weights=event_weights)[0]
    errors = numpy.sqrt(numpy.histogram(event_var, bins=var_edges, weights=event_weights**2)[0])
    return (weights, errors)


def main():
    cache_file = '.cached_analysis_info.p'
    var_edges = numpy.linspace(200, 1400, 6)

    ###################
    # LOAD EVERYTHING #
    ###################
    if len(sys.argv) > 1:
        # Load Data - Just load 3b1f Signal Region
        data_vals, data_errs = load_data(var_edges)

        # Load Background - Just use 2b Signal Region
        bgd_vals, bgd_errs = load_bgd(var_edges)

        # Load Signal - Juse use SM and then maybe cvv = 3
        sig_vals, sig_errs = load_signal(var_edges)

        data = {
            'data': (data_vals, data_errs),
            'bgd' : (bgd_vals, bgd_errs),
            'sig' : (sig_vals, sig_errs)
        }
        pickle.dump(data, open(cache_file,'wb'))
    else:
        # Load from pickle
        data = pickle.load(open(cache_file,'rb'))

    #for key, (val,err) in data.items():
    #    print(key)
    #    print(' '.join([f'{v:7.02f}' for v in val]))
    #    print(' '.join([f'{e:7.02f}' for e in err]))
    #    print()

    fig, (ax, rat) = plt.subplots(2, gridspec_kw={'height_ratios':(1,1)}, sharex=True)
    ax.errorbar(var_edges[:-1]+0.5, data['data'][0], yerr=data['data'][1], marker='.', ls='--', color='purple')
    ax.errorbar(var_edges[:-1]+0.5, data['bgd'][0], yerr=data['bgd'][1], marker='.', ls='--', color='blue')
    ax.errorbar(var_edges[:-1]+0.5, data['sig'][0], yerr=data['sig'][1], marker='.', ls='--', color='green')

    rat.errorbar(var_edges[:-1]+0.5, data['data'][0]-data['bgd'][0],
        yerr = numpy.sqrt(data['data'][1]**2 + data['bgd'][1]**2),
        marker='.', ls='--', color='red')
    rat.errorbar(var_edges[:-1]+0.5, data['sig'][0], yerr=data['sig'][1], marker='.', ls='--', color='green')
    rat.hlines(0, var_edges[0], var_edges[-1], linestyle='-', color='black')


    plt.savefig('data_dump.pdf')


    

main()
