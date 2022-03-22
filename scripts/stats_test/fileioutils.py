import pickle
import numpy
import sympy
import uproot

#_eta_cut = lambda ttree: ttree['dEta_hh'].array() < 1.5
_k2v = sympy.Symbol('\kappa_{2V}')
_kl = sympy.Symbol('\kappa_{\lambda}')
_kv = sympy.Symbol('\kappa_{V}')

_eta_cut = lambda ttree: ttree['dEta_hh'].array() > -1

full_scan_terms = [
    lambda k2v,kl,kv: kv**2 * kl**2,
    lambda k2v,kl,kv: kv**4,
    lambda k2v,kl,kv: k2v**2,
    lambda k2v,kl,kv: kv**3 * kl,
    lambda k2v,kl,kv: k2v * kl * kv,
    lambda k2v,kl,kv: kv**2 * k2v
]


def get_amplitude_function( basis_parameters, form='scalar', base_equations=full_scan_terms):
    basis_states = [ [ sympy.Rational(param) for param in basis ] for basis in basis_parameters ]
    combination_matrix = sympy.Matrix([ [ g(*base) for g in base_equations ] for base in basis_states])
    if combination_matrix.det() == 0: return None

    inversion = combination_matrix.inv()
    term_vector = sympy.Matrix([ [g(_k2v,_kl,_kv)] for g in base_equations ])
    amplitudes = sympy.Matrix([ sympy.Symbol(f'A{n}') for n in range(len(base_equations)) ])

    if form == 'vector':
        final_weight = term_vector.T * inversion
        reweight_vector_function = sympy.lambdify([_k2v, _kl, _kv], final_weight, 'numpy')
        return reweight_vector_function
    else:
        amplitudes = sympy.Matrix(
            [sympy.Symbol(f"A{n}") for n in range(len(base_equations))]
        )
        final_amplitude = (term_vector.T * inversion * amplitudes)[0]
        if form == 'expression': return final_amplitude

        amplitude_function = sympy.lambdify(
            [_k2v, _kl, _kv] + [*amplitudes], final_amplitude, "numpy"
        )
        return amplitude_function



def get_theory_combination_function():
    sm_N3LO_xsec = 1.726  # N3LO SM x-sec in fb (from Tulin's slides: https://indico.cern.ch/event/919518/contributions/3921302/attachments/2070155/3475083/LHCHHMeeting_070720.pdf)

    # Below xsec values are from AMI as of 9 December 2020, for tag: e8263_e7400_s3126_r10201
    basis_xsec_list = [
        # ([k2v, kl, kv], xsec)
        ([1, 1, 1], 1.18),
        ([1.5, 1, 1], 2.30),
        ([2, 1, 1], 9.97),
        ([1, 0, 1], 3.17),
        ([1, 10, 1], 67.4),
        ([1, 1, 1.5], 45.4),
    ]
    xsec_correction = sm_N3LO_xsec / basis_xsec_list[0][1]

    theory_basis_list, xsec_list = zip(*basis_xsec_list)
    corrected_xsecs = [xsec * xsec_correction for xsec in xsec_list]

    theory_amplitude_function = get_amplitude_function(theory_basis_list)
    theory_combination_function = lambda couplings: theory_amplitude_function( *couplings, *corrected_xsecs)
    return theory_combination_function



def load_data(var_edges, pickle_load=False):
    cache_file = '.pickled_data.p'
    if pickle_load: return pickle.load(open(cache_file,'rb'))

    data_files = [
        '/home/cmilke/Downloads/large_datasets/unblinded_data/data16_NN_100_bootstraps.root',
        '/home/cmilke/Downloads/large_datasets/unblinded_data/data17_NN_100_bootstraps.root',
        '/home/cmilke/Downloads/large_datasets/unblinded_data/data18_NN_100_bootstraps.root'
    ]
    key='m_hh'
    events = []
    for f in data_files:
        tree_name = 'sig'
        rootfile = uproot.open(f)
        ttree = rootfile[tree_name]

        pass_vbf_sel = ttree['pass_vbf_sel'].array()
        x_wt_tag = ttree['X_wt_tag'].array() > 1.5
        ntag = ttree['ntag'].array() >= 4
        eta = _eta_cut(ttree)
        valid_event = numpy.logical_and.reduce( (pass_vbf_sel, x_wt_tag, ntag, eta) )

        kinvals = ttree['m_hh'].array()[valid_event]
        weights = ttree['mc_sf'].array()[valid_event]
        lumi_weights = weights

        new_events = numpy.array([kinvals,lumi_weights])
        events.append(new_events)

    event_var, event_weights = numpy.concatenate(events, axis=1)

    weights = numpy.histogram(event_var, bins=var_edges, weights=event_weights)[0]
    errors = numpy.sqrt(numpy.histogram(event_var, bins=var_edges, weights=event_weights**2)[0])
    data = (weights, errors)
    pickle.dump(data, open(cache_file,'wb'))
    return data


def load_bgd(var_edges, pickle_load=False):
    cache_file = '.pickled_bgd.p'
    if pickle_load: return pickle.load(open(cache_file,'rb'))

    data_files = [
        (16, '/home/cmilke/Downloads/large_datasets/unblinded_data/data16_NN_100_bootstraps.root'),
        (17, '/home/cmilke/Downloads/large_datasets/unblinded_data/data17_NN_100_bootstraps.root'),
        (18, '/home/cmilke/Downloads/large_datasets/unblinded_data/data18_NN_100_bootstraps.root')
    ]
    key='m_hh'
    events = []
    for y,f in data_files:
        tree_name = 'sig'
        rootfile = uproot.open(f)
        med_norm = rootfile[f'NN_norm_bstrap_med_{y}'].value
        VR_med_norm = rootfile[f'NN_norm_VRderiv_bstrap_med_{y}'].value
        ttree = rootfile[tree_name]

        pass_vbf_sel = ttree['pass_vbf_sel'].array()
        rw_pass = ttree['rw_to_4b'].array()
        x_wt_tag = ttree['X_wt_tag'].array() > 1.5
        ntag = ttree['ntag'].array() == 2
        eta = _eta_cut(ttree)
        valid_event = numpy.logical_and.reduce( (pass_vbf_sel, rw_pass, x_wt_tag, ntag, eta) )

        kinvals = ttree['m_hh'].array()[valid_event]
        NN_weights = ttree[f'NN_d24_weight_bstrap_med_{y}'].array()[valid_event]*med_norm
        VR_weights = ttree[f'NN_d24_weight_VRderiv_bstrap_med_{y}'].array()[valid_event]*med_norm

        new_events = numpy.array([kinvals,NN_weights,VR_weights])
        events.append(new_events)

    event_var, event_weights, event_VR_weights = numpy.concatenate(events, axis=1)
    
    weights = numpy.histogram(event_var, bins=var_edges, weights=event_weights)[0]
    alt_weights = numpy.histogram(event_var, bins=var_edges, weights=event_VR_weights)[0]
    sys_errors = abs(alt_weights - weights)
    stat_errors = numpy.sqrt(numpy.histogram(event_var, bins=var_edges, weights=event_weights**2)[0])
    errors = numpy.sqrt( sys_errors**2 + stat_errors**2)
    #errors = stat_errors
    background = (weights, errors)
    pickle.dump(background, open(cache_file,'wb'))
    return background


def read_coupling_file(coupling_file=None):
    data_files = {}
    with open(coupling_file) as coupling_list:
        for line in coupling_list:
            if line.strip().startswith('#'): continue
            linedata = line.split()
            couplings = tuple([ float(p) for p in linedata[:3] ])
            data_file = linedata[3]
            if couplings not in data_files:
                data_files[couplings] = [data_file]
            else:
                data_files[couplings].append(data_file)
    return data_files


def extract_ntuple_events(ntuple, key=None, tree_name=None):
    #tree_name = 'sig_highPtcat'
    tree_name = 'sig'

    rootfile = uproot.open(ntuple)
    #DSID = rootfile['DSID']._members['fVal']
    #nfiles = 1
    #while( DSID / nfiles > 600050 ): nfiles += 1
    #DSID = int(DSID / nfiles)
    #print(ntuple, DSID)
    ttree = rootfile[tree_name]

    #if tree_name == 'sig':
    #if True:
    if False:
        kinvals = ttree['m_hh'].array()
        weights = ttree['mc_sf'].array()[:,0]
        run_number = ttree['run_number'].array()
    else: # Selections
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

    events = numpy.array([kinvals,lumi_weights])
    return events


def get_events(parameter_list, data_files, reco=True):
    events_list = []
    for couplings in parameter_list:
        new_events = []
        for f in data_files[couplings]:
            new_events.append( extract_ntuple_events(f,key='m_hh') )
        events = numpy.concatenate(new_events, axis=1)
        events_list.append(events)
    return events_list


def retrieve_reco_weights(var_edges, reco_events):
    reco_weights = numpy.histogram(reco_events[0], bins=var_edges, weights=reco_events[1])[0]
    reco_errors = numpy.sqrt(numpy.histogram(reco_events[0], bins=var_edges, weights=reco_events[1]**2)[0])
    return [reco_weights, reco_errors]


def reco_reweight(reweight_vector_function, coupling_parameters, base_weights, base_errors):
    multiplier_vector = reweight_vector_function(*coupling_parameters)[0]
    reweighted_weights = numpy.array([ w*m for w,m in zip(base_weights, multiplier_vector) ])
    linearly_combined_weights = reweighted_weights.sum(axis=0)
    reweighted_errors2 = numpy.array([ (w*m)**2 for w,m in zip(base_errors, multiplier_vector) ])
    linearly_combined_errors = numpy.sqrt( reweighted_errors2.sum(axis=0) )
    return linearly_combined_weights, linearly_combined_errors


def load_signal_basis(var_edges, pickle_load=False):
    base_couplings = [ (1.0, 1.0, 1.0), (1.5, 1.0, 1.0), (1.0, 2.0, 1.0), (1.0, 10.0, 1.0), (1.0, 1.0, 0.5), (1.0, -5.0, 0.5) ]
    cache_file = '.pickled_sig.p'
    if pickle_load: 
        return pickle.load(open(cache_file,'rb'))
    else:
        data_files = read_coupling_file('/home/cmilke/Documents/dihiggs/coupling_scan/basis_files/nnt_coupling_file_2021Aug_test.dat')
        basis_events = get_events(base_couplings, data_files)
        basis_data = [ retrieve_reco_weights(var_edges, events) for events in basis_events ]
        basis_collection = (base_couplings, *list(zip(*basis_data)))
        pickle.dump(basis_collection, open(cache_file,'wb'))
        return basis_collection


def load_signal(var_edges, pickle_load=False):
    base_couplings, basis_weights, basis_errors = load_signal_basis(var_edges, pickle_load=pickle_load)
    reweight_vector_function = get_amplitude_function(base_couplings, form='vector', base_equations=full_scan_terms)
    signal = lambda couplings: reco_reweight(reweight_vector_function, couplings, basis_weights, basis_errors)
    return signal



def load_ggF_bgd(var_edges):
    signal_files = [ # SM
        '/home/cmilke/Downloads/large_datasets/nano_ntuples/ggF_may21/600463_mc16a/hh_600463_mc16a.root',
        '/home/cmilke/Downloads/large_datasets/nano_ntuples/ggF_may21/600463_mc16d/hh_600463_mc16d.root',
        '/home/cmilke/Downloads/large_datasets/nano_ntuples/ggF_may21/600463_mc16e/hh_600463_mc16e.root'
    ]
    #signal_files = [ # k2v=3
    #    '/home/cmilke/Documents/dihiggs/nano_ntuples/2021_aug_test/VBF_nonres2021_502975_MC16a-2015-2016_NanoNTuple.root',
    #    '/home/cmilke/Documents/dihiggs/nano_ntuples/2021_aug_test/VBF_nonres2021_502975_MC16d-2017_NanoNTuple.root',
    #    '/home/cmilke/Documents/dihiggs/nano_ntuples/2021_aug_test/VBF_nonres2021_502975_MC16e-2018_NanoNTuple.root'
    #]

    key='m_hh'
    events = []
    for f in signal_files:
        tree_name = 'sig'
        rootfile = uproot.open(f)
        ttree = rootfile[tree_name]

        pass_vbf_sel = ttree['pass_vbf_sel'].array()
        x_wt_tag = ttree['X_wt_tag'].array() > 1.5
        ntag = ttree['ntag'].array() >= 4
        eta = _eta_cut(ttree)
        valid_event = numpy.logical_and.reduce( (pass_vbf_sel, x_wt_tag, ntag, eta) )

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
