#!/usr/bin/python
import sys
import math
import random
import statistics
import pickle
import scipy
import scipy.stats
import sympy
import numpy
import uproot
from matplotlib import pyplot as plt
import pyhf


#_eta_cut = lambda ttree: ttree['dEta_hh'].array() < 1.5
_eta_cut = lambda ttree: ttree['dEta_hh'].array() > -1

full_scan_terms = [
    lambda k2v,kl,kv: kv**2 * kl**2,
    lambda k2v,kl,kv: kv**4,
    lambda k2v,kl,kv: k2v**2,
    lambda k2v,kl,kv: kv**3 * kl,
    lambda k2v,kl,kv: k2v * kl * kv,
    lambda k2v,kl,kv: kv**2 * k2v
]


def log_poisson(n,v):
    lp = numpy.full(len(n),-v).astype(float)
    nz = n!=0
    n = n[nz]
    lp[nz] = n - v + n*numpy.log(v/n) - (1/2)*numpy.log(2*math.pi*n)
    return lp
    

def load_data(var_edges):
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
    return (weights, errors)


def load_bgd(var_edges):
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
    return (weights, errors)


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


def get_amplitude_function( basis_parameters, as_scalar=True, base_equations=full_scan_terms):
    _k2v = sympy.Symbol('\kappa_{2V}')
    _kl = sympy.Symbol('\kappa_{\lambda}')
    _kv = sympy.Symbol('\kappa_{V}')

    basis_states = [ [ sympy.Rational(param) for param in basis ] for basis in basis_parameters ]
    combination_matrix = sympy.Matrix([ [ g(*base) for g in base_equations ] for base in basis_states])
    if combination_matrix.det() == 0: return None

    inversion = combination_matrix.inv()
    term_vector = sympy.Matrix([ [g(_k2v,_kl,_kv)] for g in base_equations ])
    amplitudes = sympy.Matrix([ sympy.Symbol(f'A{n}') for n in range(len(base_equations)) ])

    if as_scalar:
        # FYI, numpy outputs a 1x1 matrix here, so I use the [0] to get just the equation
        final_amplitude = (term_vector.T*inversion*amplitudes)[0]
        amplitude_function = sympy.lambdify([_k2v, _kl, _kv], final_amplitude, 'numpy')
        return amplitude_function
    else:
        final_weight = term_vector.T * inversion
        reweight_vector_function = sympy.lambdify([_k2v, _kl, _kv], final_weight, 'numpy')
        return reweight_vector_function


def reco_reweight(reweight_vector_function, coupling_parameters, base_weights, base_errors):
    multiplier_vector = reweight_vector_function(*coupling_parameters)[0]
    reweighted_weights = numpy.array([ w*m for w,m in zip(base_weights, multiplier_vector) ])
    linearly_combined_weights = reweighted_weights.sum(axis=0)
    reweighted_errors2 = numpy.array([ (w*m)**2 for w,m in zip(base_errors, multiplier_vector) ])
    linearly_combined_errors = numpy.sqrt( reweighted_errors2.sum(axis=0) )
    return linearly_combined_weights, linearly_combined_errors


def load_signal(var_edges):
    base_couplings = [ (1.0, 1.0, 1.0), (1.5, 1.0, 1.0), (1.0, 2.0, 1.0), (1.0, 10.0, 1.0), (1.0, 1.0, 0.5), (1.0, -5.0, 0.5) ]
    data_files = read_coupling_file('/home/cmilke/Documents/dihiggs/coupling_scan/basis_files/nnt_coupling_file_2021Aug_test.dat')
    basis_events = get_events(base_couplings, data_files)
    basis_data = [ retrieve_reco_weights(var_edges, events) for events in basis_events ]
    basis_weights, basis_errors = list(zip(*basis_data))
    reweight_vector_function = get_amplitude_function(base_couplings, as_scalar=False, base_equations=full_scan_terms)
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


def make_data_display_plots(data=None, var_edges=None):
    fig, (ax, rat) = plt.subplots(2, gridspec_kw={'height_ratios':(2,1)}, sharex=True)
    ax.errorbar(var_edges[:-1]+0.5, data['data'][0], yerr=data['data'][1], marker='.', ls='--', color='purple', label='Data')
    ax.errorbar(var_edges[:-1]+0.5, data['bgd'][0], yerr=data['bgd'][1], marker='.', ls='--', color='blue', label='Bgd')
    ax.errorbar(var_edges[:-1]+0.5, data['sig'][0], yerr=data['sig'][1], marker='.', ls='--', color='green', label='Signal')

    sensitivity = data['sig'][0] / numpy.sqrt( data['data'][1]**2 + data['bgd'][1]**2)
    rat.errorbar(var_edges[:-1]+0.5, sensitivity,
        marker='.', ls='--', color='red')
    rat.hlines(0, var_edges[0], var_edges[-1], linestyle='-', color='black')

    ax.legend()


    plt.savefig('data_dump.pdf')
    plt.close()


def make_basic_poisson_plots(data=None, prefix=''):
    def log_poisson(n,v):
        lp = numpy.full(len(n),-v).astype(float)
        nz = n!=0
        n = n[nz]
        lp[nz] = n - v + n*numpy.log(v/n) - (1/2)*numpy.log(2*math.pi*n)
        return lp
    


    signal = sum(data['sig'][0])
    background = sum(data['bgd'][0])
    expectation = signal+background
    observed = int(sum(data['data'][0]))
    max_n = int(expectation)*2
    poisson_inputs = numpy.arange(0,max_n,1)
    log_poisson_values = log_poisson(poisson_inputs,expectation)
    poisson_values = numpy.exp(log_poisson_values)
    cumulative_poisson = poisson_values[::-1].cumsum()[::-1]
    pvalue = cumulative_poisson[observed]


    fig, ax = plt.subplots()
    ax.plot(poisson_inputs, poisson_values, label='Poisson PDF')
    ax.axvline(observed, ls='--', label=f'Observed n={observed}', color='red')
    ax.fill_between(range(observed,max_n), 0, poisson_values[observed:], color='green', hatch='///', alpha=0.3, label=f'p-value={pvalue:.2f}')
    plt.xlabel('Number of Events')
    plt.ylabel('Probability')
    plt.xlim(expectation*.9, expectation*1.1)
    plt.ylim(0)
    ax.legend()
    plt.savefig(prefix+'_poisson.pdf')
    plt.close()

    fig, ax = plt.subplots()
    ax.plot(poisson_inputs, cumulative_poisson, label='Poisson Cumulative PDF')
    ax.axvline(observed, ls='--', label=f'Observed n={observed}', color='red')
    ax.axhline(pvalue, ls='dotted', label=f'p-value={pvalue:.2f}', color='green')
    plt.xlabel('Number of Events')
    plt.ylabel('Cumulative Probability')
    plt.xlim(expectation*.9, expectation*1.1)
    plt.ylim(0)
    ax.legend()
    plt.savefig(prefix+'_Cpoisson.pdf')
    plt.close()


def make_sb_poisson_plots(results=None, prefix='', couplings=None):
    mu = 3
    signal = mu*sum(results['sig'](couplings)[0])
    background = int(sum(results['bgd'][0]))
    expectation = signal+background
    observed = int(sum(results['data'][0]))
    max_n = int(expectation)*4 # This is basically just the range of the plots
    poisson_inputs = numpy.arange(0,max_n,1)
    log_poisson_bgd_values = log_poisson(poisson_inputs,background)
    log_poisson_bs_values = log_poisson(poisson_inputs,expectation)
    poisson_bgd_values = numpy.exp(log_poisson_bgd_values)
    poisson_bs_values = numpy.exp(log_poisson_bs_values)
    poisson_bgd_values = poisson_bgd_values / poisson_bgd_values.sum()
    poisson_bs_values = poisson_bs_values / poisson_bs_values.sum()
    cumulative_bgd_poisson = poisson_bgd_values[::-1].cumsum()[::-1]
    cumulative_bs_poisson = poisson_bs_values[::-1].cumsum()[::-1]
    #cumulative_bgd_poisson = poisson_bgd_values.cumsum()
    #cumulative_bs_poisson = poisson_bs_values.cumsum()
    print(cumulative_bgd_poisson)
    print(cumulative_bgd_poisson)
    cumulative_sig_poisson = cumulative_bs_poisson / (1 - cumulative_bgd_poisson)
    pvalue = cumulative_sig_poisson[observed]
    bgd_pvalue = cumulative_bgd_poisson[observed]
    bs_pvalue = cumulative_bs_poisson[observed]
    print(bs_pvalue)


    fig, ax = plt.subplots()
    ax.plot(poisson_inputs, poisson_bgd_values, label='B PDF')
    ax.plot(poisson_inputs, poisson_bs_values, ls='--', label='S+B PDF')
    ax.axvline(observed, ls='--', label=f'Observed n={observed}', color='red')
    ax.fill_between(range(observed,max_n), 0, poisson_bgd_values[observed:], color='green', hatch='///', alpha=0.3, label=f'bgd p-value={bgd_pvalue:.2f}')
    plt.xlabel('Number of Events')
    plt.ylabel('Probability')
    plt.xlim(expectation*.5, expectation*1.5)
    plt.ylim(0)
    ax.legend()
    plt.savefig(prefix+'_poisson.pdf')
    plt.close()

    fig, ax = plt.subplots()
    ax.plot(poisson_inputs, cumulative_bgd_poisson, label='B C-PDF')
    ax.plot(poisson_inputs, cumulative_bs_poisson, label='S+B C-PDF', ls='--')
    ax.plot(poisson_inputs, cumulative_sig_poisson, label='S+B/(1-B) C-PDF', color='purple')
    ax.axvline(observed, ls='--', label=f'Observed n={observed}', color='red')
    ax.axhline(pvalue, ls='dotted', label=f'Signal p-value={pvalue:.2f}', color='magenta')
    ax.axhline(bgd_pvalue, ls='dotted', label=f'Background p-value={bgd_pvalue:.2f}', color='green')
    plt.xlabel('Number of Events')
    plt.ylabel('Cumulative Probability')
    plt.xlim(expectation*.5, expectation*1.5)
    plt.ylim(0,1)
    ax.legend()
    plt.savefig(prefix+'_Cpoisson.pdf')
    plt.close()


def derive_maximum_likelihood_mu(results=None, couplings=None):
    data_vals, data_errs = results['data']
    sig_vals, sig_errs = results['sig']
    bgd_vals, bgd_errs = results['bgd']

    mu = sympy.Symbol('mu')
    #sig_vector = [ sympy.Symbol(f's{i}') for i in range(len(sig_vals)) ]
    #bgd_vector = [ sympy.Symbol(f'b{i}') for i in range(len(bgd_vals)) ]
    #events_observed = sympy.Symbol('n')
    theoretical_events = lambda s,b: mu*s+b
    llh = lambda n,T: (n-1)*sympy.log(T) - T - n*sympy.log(n) + n
    ellh = lambda n,events: llh(n,sum(events)) + sum([sympy.log(e) for e in events])

    theoretical_events_per_bin = [ theoretical_events(sig, bgd) for sig,bgd in zip(sig_vals, bgd_vals) ]
    extended_log_likelihood_function = ellh(sum(data_vals), theoretical_events_per_bin)
    #theoretical_events_per_bin = [ theoretical_events(s, b) for s,b in zip(sig_vector, bgd_vector) ]
    #extended_log_likelihood_function = ellh(events_observed, theoretical_events_per_bin)

    ell_derivative = sympy.diff(extended_log_likelihood_function, mu)

    zero = sympy.solvers.solvers.nsolve(ell_derivative, mu, 200)


def make_lazy_mu_probability_distro(results=None, couplings=None):
    data_yield = int(sum(results['data'][0]))
    sig_yield = sum(results['sig'](couplings)[0])
    bgd_yield = sum(results['bgd'][0])

    #mu = sympy.Symbol('mu')
    #theoretical_events = lambda s,b: mu*s+b
    #poisson = lambda n,k: k**n * sympy.exp(-k) / ( (n/math.e)**n * sympy.sqrt(2*math.pi*n) )
    #sympy_log_poisson = lambda n,v: n - v + n*sympy.log(v/n) - (1/2)*sympy.log(2*math.pi*n) if n > 0 else -v


    def get_pvalue(signal, background, observed): 
        expectation = signal+background
        max_n = int(expectation)*4 # This is basically just the range of the plots
        poisson_inputs = numpy.arange(0,max_n,1)
        log_poisson_bgd_values = log_poisson(poisson_inputs,background)
        log_poisson_bs_values = log_poisson(poisson_inputs,expectation)
        poisson_bgd_values = numpy.exp(log_poisson_bgd_values)
        poisson_bs_values = numpy.exp(log_poisson_bs_values)
        poisson_bgd_values = poisson_bgd_values / poisson_bgd_values.sum()
        poisson_bs_values = poisson_bs_values / poisson_bs_values.sum()
        cumulative_bgd_poisson = poisson_bgd_values[::-1].cumsum()[::-1]
        cumulative_bs_poisson = poisson_bs_values[::-1].cumsum()[::-1]
        cumulative_sig_poisson = cumulative_bs_poisson / (1 - cumulative_bgd_poisson)
        pvalue = cumulative_sig_poisson[observed]
        return 1-pvalue


    def get_cumlative_PDF(max_mu=1000, zoom_passes=2, precision_threshold=1e-5):
        for i in range(zoom_passes+1):
            mu_values = numpy.linspace(0,max_mu,100)
            pvalues = [ get_pvalue(mu*sig_yield, bgd_yield, data_yield) for mu in mu_values ]
            pvalues = numpy.array(pvalues)

            #print(mu_values)
            #print(mu_probabilities)
            #print()
            #mu_probabilities /= mu_probabilities.sum()
            #cumalitive_mu_PDF = mu_probabilities[::-1].cumsum()[::-1]
            max_mu = mu_values[numpy.argmax(pvalues<precision_threshold)]
        return mu_values, pvalues
    mu_values, pvalues = get_cumlative_PDF()
    exclusion_limit_index = numpy.argmax(pvalues < 0.05)
    mu_limit = mu_values[exclusion_limit_index]


    #fig, ax = plt.subplots()
    #ax.errorbar(mu_values, mu_probabilities, color='black', ls='--')
    #ax.axvline(x=mu_limit)
    #plt.savefig('mu_pdf.pdf')
    #plt.close()

    fig, ax = plt.subplots()
    ax.errorbar(mu_values, pvalues, color='black', ls='--')
    ax.hlines(0.05, mu_values[0], mu_values[-1], color='red', ls='dotted')
    ax.axvline(x=mu_limit)
    plt.savefig('mu_Cpdf.pdf')
    plt.close()


def get_single_mu_pval(results=None, couplings=None, mu_val=None):
    data_vals, data_errs = results['data']
    sig_vals, sig_errs = results['sig']
    bgd_vals, bgd_errs = results['bgd']

    def log_poisson(n,v):
        lp = -v.astype(float)
        nz = numpy.logical_and(n!=0,v!=0)
        n,v = n[nz], v[nz]
        lp[nz] = n - v + n*numpy.log(v/n) - (1/2)*numpy.log(2*math.pi*n)
        return lp

    def log_gauss(n,v,sigma):
        lg = -(1/2)*( (n-v)/sigma )**2 - numpy.log(sigma) - (1/2)*numpy.log(2*math.pi)
        return lg

    def expectation(mu, theta, sig, bgd):
        expectation_vals = mu * sig_vals + theta*bgd_vals
        expectation_vals[ expectation_vals < 0 ] = 0
        return expectation_vals

    def loglikelihood(n, mu, theta, sigma):
        expec_val = expectation(mu, theta, sig_vals, bgd_vals)
        poisson = log_poisson(n,expec_val).sum()
        gauss = log_gauss(n, expec_val, sigma).sum()
        L = poisson+gauss
        return L

    test_stat = lambda L,L_max,mu_max: -2*(L-L_max) if mu_max < mu_val else 0


    def get_no_nuissance_L(n,mu):
        try:
            minimizeable_fcn = lambda theta: -loglikelihood(n,mu,theta,bgd_errs)
            max_theta = scipy.optimize.minimize(minimizeable_fcn, 1).x[0]
            L = loglikelihood(n,mu,max_theta,bgd_errs)
            return L
        except scipy.optimize.linesearch.LineSearchWarning:
            return None

    def get_fully_maximized_L(n):
        #print(' | '.join([f'{mu:8.1f}, {numpy.exp(loglikelihood(n,mu)):4.1f}' for mu in numpy.linspace(-1,1,10)]))
        #print(' | '.join([f'{mu:8.1f}, {numpy.exp(loglikelihood(n,mu)):4.1f}' for mu in numpy.linspace(-10,10,10)]))
        #print(' | '.join([f'{mu:8.1f}, {numpy.exp(loglikelihood(n,mu)):4.1f}' for mu in numpy.linspace(-100,100,10)]))
        #print(' | '.join([f'{mu:8.1f}, {numpy.exp(loglikelihood(n,mu)):4.1f}' for mu in numpy.linspace(-1000,1000,10)]))
        try:
            fully_minimizeable_fcn = lambda minvars: -loglikelihood(n,minvars[0],minvars[1],bgd_errs)
            maximals = scipy.optimize.minimize(fully_minimizeable_fcn, [1,1]).x
            max_mu = maximals[0]
            max_theta = maximals[1]

            #max_mu = scipy.optimize.minimize(lambda mu: -numpy.exp(loglikelihood(n,mu)), 5).x[0]
            if max_mu < 0: max_mu = 0
            max_L = loglikelihood(n,max_mu,max_theta,bgd_errs)
            return max_L, max_mu
        except scipy.optimize.linesearch.LineSearchWarning:
            return None



    num_toy_distros = 1000
    toy_test_stat_values = []
    for toy_index in range(num_toy_distros):
        toy_sig = numpy.random.poisson(mu_val*sig_vals)
        toy_bgd = numpy.random.poisson(bgd_vals)
        toy_distro = toy_sig+toy_bgd

        base_L = get_no_nuissance_L(toy_distro,mu_val)
        max_vals = get_fully_maximized_L(toy_distro)
        if max_vals is None: continue
        test_stat_val = test_stat(base_L, *max_vals)
        toy_test_stat_values.append(test_stat_val)
    toy_test_stat_values.sort()
    observed_max = get_fully_maximized_L(data_vals)
    observed_test_stat_value = test_stat(loglikelihood(data_vals,mu_val), *observed_max)
    p_value = (toy_test_stat_values > observed_test_stat_value).sum() / len(toy_test_stat_values)
    return p_value


def make_mu_probability_distro(results=None, couplings=None):
    data_vals, data_errs = results['data']
    sig_vals, sig_errs = results['sig']
    bgd_vals, bgd_errs = results['bgd']

    def get_sig_pval(mu):
        bgd_sig_pval = get_single_mu_pval(data=results, couplings=couplings, mu_val=mu)
        bgd_only_pval = get_single_mu_pval(data=results, couplings=couplings, mu_val=0)
        pval = 0
        if bgd_only_pval < 1: pval = bgd_sig_pval/(1-bgd_only_pval)
        return pval
        

    #mu_values = numpy.linspace(0,1000,10)
    mu_values = [1,2,3,4]
    p_values = []
    for mu in mu_values:
        pval = get_sig_pval(mu)

        model = pyhf.simplemodels.uncorrelated_background(signal=list(sig_vals), bkg=list(bgd_vals), bkg_uncertainty=list(bgd_errs))
        pyhf_data = pyhf.tensorlib.astensor(list(data_vals)+model.config.auxdata)
        #pyhf.infer.hypotest(mu, list(data_vals), model, init_pars=inits, par_bounds=bounds, return_expected_set=True, test_stat="qtilde",)
        #CLs_obs, CLs_exp_band = pyhf.infer.hypotest(mu, pyhf_data, model, return_expected_set=True, test_stat="qtilde",par_bounds=[[0,500]]*(len(sig_vals)+1))
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            CLs_obs, CLs_exp_band = pyhf.infer.hypotest(mu, pyhf_data, model, return_expected_set=True, test_stat="qtilde")
        #print(CLs_obs)
        print(pval, CLs_exp_band[2])


    exit()

    #exclusion_limit_index = numpy.argmax(cumalitive_mu_PDF < 0.05)
    mu_limit = mu_values[exclusion_limit_index]

    fig, ax = plt.subplots()
    ax.errorbar(mu_values, p_values, color='black', ls='--')
    plt.savefig('mu_pval_test.pdf')
    plt.close()




def main():
    cache_file = '.cached_analysis_info.p'
    var_edges = numpy.linspace(200, 1400, 30)

    ###################
    # LOAD EVERYTHING #
    ###################
    if len(sys.argv) < 2:
        # Load Signal - Just use SM and then maybe cvv = 3
        signal = load_signal(var_edges)

        # Load ggF Background
        ggfB_vals, ggfB_errs = load_ggF_bgd(var_edges)

        # Load Data - Just load 3b1f Signal Region
        data_vals, data_errs = load_data(var_edges)

        # Load Background - Just reweight 2b Signal Region
        bgd_vals, bgd_errs = load_bgd(var_edges)

        results = {
            'data': (data_vals, data_errs),
            'bgd' : (bgd_vals, bgd_errs),
            'sig' : signal,
            'ggfB' : (ggfB_vals, ggfB_errs)
        }
        #pickle.dump(results, open(cache_file,'wb'))
    else:
        # Load from pickle
        results = pickle.load(open(cache_file,'rb'))

    print('Data Loaded')

    #data = {
    #    'data': (numpy.array([11]), numpy.array([1])),
    #    'sig' : (numpy.array([3]), numpy.array([1])),
    #    'bgd' : (numpy.array([8]), numpy.array([5])),
    #}
    #make_basic_poisson_plots(results=results, prefix='toy')


    #for key, (val,err) in results.items():
    #    print(key)
    #    print(' '.join([f'{v:7.02f}' for v in val]))
    #    print(' '.join([f'{e:7.02f}' for e in err]))
    #    print()
    #sig_vals[sig_vals==0] = 1e-5
    #bgd_vals[bgd_vals==0] = 1e-5

    #print('Signal')
    #print(results['sig'][0].sum(), results['sig'][1].sum())
    #print()
    #print('ggF Bgd')
    #print(results['ggfB'][0].sum(), results['ggfB'][1].sum())
    #print()
    #print('Bgd')
    #print(results['bgd'][0].sum(), results['bgd'][1].sum())
    #print()
    #print('Data')
    #print(results['data'][0].sum(), results['data'][1].sum())

    ##make_data_display_plots(results=results,var_edges=var_edges)
    make_sb_poisson_plots(results=results, prefix='total_yield', couplings=(1,1,1))
    make_lazy_mu_probability_distro(results=results, couplings=(1,1,1))
    #make_mu_probability_distro(results=results, couplings=(1,1,1))
    #derive_maximum_likelihood_mu(results=results, couplings=(1,1,1))




    

main()
