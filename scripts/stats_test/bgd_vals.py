#!/usr/bin/python
import sys
import numpy
import uproot


_eta_cut = lambda ttree: ttree['dEta_hh'].array() > -1
_mhh_cut_lo = lambda ttree: ttree['m_hh'].array() > 0
_mhh_cut_hi = lambda ttree: ttree['m_hh'].array() < 5000


def get_data(tree_name, btag):
    data_files = [
        '/home/cmilke/Downloads/large_datasets/unblinded_data/data16_NN_100_bootstraps.root',
        '/home/cmilke/Downloads/large_datasets/unblinded_data/data17_NN_100_bootstraps.root',
        '/home/cmilke/Downloads/large_datasets/unblinded_data/data18_NN_100_bootstraps.root'
    ]

    event_yield = 0
    for f in data_files:
        rootfile = uproot.open(f)
        ttree = rootfile[tree_name]

        pass_vbf_sel = ttree['pass_vbf_sel'].array()
        x_wt_tag = ttree['X_wt_tag'].array() > 1.5
        if btag == 2:
            ntag = ttree['ntag'].array() == 2
        elif btag == 4:
            ntag = ttree['ntag'].array() >= 4
        eta = _eta_cut(ttree)
        mhh_lo = _mhh_cut_lo(ttree)
        mhh_hi = _mhh_cut_hi(ttree)
        valid_event = numpy.logical_and.reduce( (pass_vbf_sel, x_wt_tag, ntag, eta, mhh_lo, mhh_hi) )
        weights = ttree['mc_sf'].array()[valid_event]
        event_yield += sum(weights)
    return event_yield

def get_mc(tree_name, btag):
    sig_files = [
        '/home/cmilke/Documents/dihiggs/nano_ntuples/2021_aug_test/VBF_nonres2021_502971_MC16a-2015-2016_NanoNTuple.root',
        '/home/cmilke/Documents/dihiggs/nano_ntuples/2021_aug_test/VBF_nonres2021_502971_MC16d-2017_NanoNTuple.root',
        '/home/cmilke/Documents/dihiggs/nano_ntuples/2021_aug_test/VBF_nonres2021_502971_MC16e-2018_NanoNTuple.root'

        #'/home/cmilke/Documents/dihiggs/nano_ntuples/2021_aug_test/VBF_nonres2021_502970_MC16a-2015-2016_NanoNTuple.root',
        #'/home/cmilke/Documents/dihiggs/nano_ntuples/2021_aug_test/VBF_nonres2021_502970_MC16d-2017_NanoNTuple.root',
        #'/home/cmilke/Documents/dihiggs/nano_ntuples/2021_aug_test/VBF_nonres2021_502970_MC16e-2018_NanoNTuple.root'
    ]

    event_yield = 0
    for f in sig_files:
        rootfile = uproot.open(f)
        ttree = rootfile[tree_name]

        pass_vbf_sel = ttree['pass_vbf_sel'].array()
        x_wt_tag = ttree['X_wt_tag'].array() > 1.5
        if btag == 2:
            ntag = ttree['ntag'].array() == 2
        elif btag == 4:
            ntag = ttree['ntag'].array() >= 4
        eta = _eta_cut(ttree)
        mhh_lo = _mhh_cut_lo(ttree)
        mhh_hi = _mhh_cut_hi(ttree)
        valid_event = numpy.logical_and.reduce( (pass_vbf_sel, x_wt_tag, ntag, eta, mhh_lo, mhh_hi) )

        weights = ttree['mc_sf'].array()[:,0][valid_event]
        run_number = ttree['run_number'].array()[valid_event]

        #mc2015 = ( run_number < 296939 ) * 3.2
        mc2015 = ( run_number < 296939 ) * 0
        mc2016 = ( numpy.logical_and(296939 < run_number, run_number < 320000) ) * 24.6
        mc2017 = ( numpy.logical_and(320000 < run_number, run_number < 350000) ) * 43.65
        mc2018 = ( numpy.logical_and(350000 < run_number, run_number < 370000) ) * 58.45
        all_years = mc2015 + mc2016 + mc2017 + mc2018
        lumi_weights = weights * all_years
        event_yield += sum(lumi_weights)
    return event_yield




def main():
    data_yields = {}
    data_yields['2b control'] = get_data('control',2)
    data_yields['2b signal'] = get_data('sig',2)
    data_yields['4b control'] = get_data('control',4)
    data_yields['4b signal'] = data_yields['4b control'] * (data_yields['2b signal'] / data_yields['2b control'])
    data_yields['observed'] = get_data('sig',4)

    data_yields['mc 2b control'] = get_mc('control',2)
    data_yields['mc 2b signal'] = get_mc('sig',2)
    data_yields['mc 4b control'] = get_mc('control',4)
    data_yields['mc 4b signal'] = get_mc('sig',4)


    for k,v in data_yields.items(): print(k, int(v))
    mc_ratio_2bcontrol = data_yields['mc 2b control'] / data_yields['2b control']
    mc_ratio_2bsig = data_yields['mc 2b signal'] / data_yields['2b signal']
    mc_ratio_4bcontrol = data_yields['mc 4b control'] / data_yields['4b control']
    mc_ratio_4bsig = data_yields['mc 4b signal'] / data_yields['4b signal']
    print()
    print(f'2b control {mc_ratio_2bcontrol*1e3}')
    print(f'2b sig {mc_ratio_2bsig*1e3}')
    print(f'4b control {mc_ratio_4bcontrol*1e3}')
    print(f'4b sig {mc_ratio_4bsig*1e3}')
    print(f'control ratio {mc_ratio_4bcontrol/mc_ratio_2bcontrol}')
    print(f'signal ratio {mc_ratio_4bsig/mc_ratio_2bsig}')



main()
