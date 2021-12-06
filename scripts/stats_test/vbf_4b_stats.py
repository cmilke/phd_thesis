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
    signal_files = [
        '/home/cmilke/Documents/dihiggs/nano_ntuples/2021_aug_test/VBF_nonres2021_502970_MC16a-2015-2016_NanoNTuple.root',
        '/home/cmilke/Documents/dihiggs/nano_ntuples/2021_aug_test/VBF_nonres2021_502970_MC16d-2017_NanoNTuple.root',
        '/home/cmilke/Documents/dihiggs/nano_ntuples/2021_aug_test/VBF_nonres2021_502970_MC16e-2018_NanoNTuple.root'
    ]
    #signal_files = [
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


def load_fake_bgd(data_vals, data_errs, sig_vals, sig_errs):
    print()
    print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
    print('!!!    LOADING FAKE BACKGROUND    !!!')
    print('!!! ALL RESULTS ARE COMPLETE LIES !!!')
    print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
    print()

    #ideal_bgd = data_vals-sig_vals # to make Signal compatible with data
    #ideal_bgd = data_vals # to make signal incompatible with data
    ideal_bgd = data_vals*1.1 # to make signal very incompatible with data
    fake_errs = data_errs
    jitter_amount = 0.005 # Decrease this to increase result significance
    jittered_fake_bgd = scipy.stats.norm.rvs(loc=ideal_bgd, scale=ideal_bgd*jitter_amount)
    return (jittered_fake_bgd, fake_errs*jitter_amount)


def make_data_display_plots(data=None, var_edges=None):
    fig, (ax, rat) = plt.subplots(2, gridspec_kw={'height_ratios':(1,1)}, sharex=True)
    ax.errorbar(var_edges[:-1]+0.5, data['data'][0], yerr=data['data'][1], marker='.', ls='--', color='purple', label='Data')
    ax.errorbar(var_edges[:-1]+0.5, data['bgd'][0], yerr=data['bgd'][1], marker='.', ls='--', color='blue', label='Bgd')
    ax.errorbar(var_edges[:-1]+0.5, data['sig'][0], yerr=data['sig'][1], marker='.', ls='--', color='green', label='Signal')

    rat.errorbar(var_edges[:-1]+0.5, data['data'][0]-data['bgd'][0],
        yerr = numpy.sqrt(data['data'][1]**2 + data['bgd'][1]**2),
        marker='.', ls='--', color='red')
    rat.errorbar(var_edges[:-1]+0.5, data['sig'][0], yerr=data['sig'][1], marker='.', ls='--', color='green')
    rat.hlines(0, var_edges[0], var_edges[-1], linestyle='-', color='black')

    ax.legend()


    plt.savefig('data_dump.pdf')
    plt.close()


def derive_maximum_likelihood_mu(data=None, couplings=None):
    data_vals, data_errs = data['data']
    sig_vals, sig_errs = data['sig']
    bgd_vals, bgd_errs = data['bgd']

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
    print(extended_log_likelihood_function)
    print()

    ell_derivative = sympy.diff(extended_log_likelihood_function, mu)
    print(ell_derivative)
    print()

    zero = sympy.solvers.solvers.nsolve(ell_derivative, mu, 200)
    print(zero)


def make_lazy_mu_probability_distro(data=None, couplings=None):
    data_vals, data_errs = data['data']
    sig_vals, sig_errs = data['sig']
    bgd_vals, bgd_errs = data['bgd']

    mu = sympy.Symbol('mu')
    theoretical_events = lambda s,b: mu*s+b
    #poisson = lambda n,k: k**n * sympy.exp(-k) / ( (n/math.e)**n * sympy.sqrt(2*math.pi*n) )
    log_poisson = lambda n,k: n - k + n*sympy.log(k/n) if n > 0 else -k

    theory_vals = [ theoretical_events(sig, bgd) for sig,bgd in zip(sig_vals, bgd_vals) ]
    loglikelihood = sum( [ log_poisson(data,theory) for data, theory in zip(data_vals, theory_vals) ] )
    log_prior = 0 # log(1); Uniform Prior, "Principle of Ignorance"
    log_probability_of_muXprobability_of_data = loglikelihood+log_prior

    mu_log_probability_function = sympy.lambdify(mu, log_probability_of_muXprobability_of_data, 'numpy')

    def get_cumlative_PDF(max_mu=1000, zoom_passes=2, precision_threshold=1e-5):
        for i in range(zoom_passes+1):
            mu_values = numpy.linspace(0,max_mu,100)
            mu_log_probabilities = mu_log_probability_function(mu_values)
            mu_probabilities = numpy.exp(mu_log_probabilities)
            mu_probabilities /= mu_probabilities.sum()
            cumalitive_mu_PDF = mu_probabilities[::-1].cumsum()[::-1]
            max_mu = mu_values[numpy.argmax(cumalitive_mu_PDF<precision_threshold)]
        return mu_values, mu_probabilities, cumalitive_mu_PDF
    mu_values, mu_probabilities, cumalitive_mu_PDF = get_cumlative_PDF()
    exclusion_limit_index = numpy.argmax(cumalitive_mu_PDF < 0.05)
    mu_limit = mu_values[exclusion_limit_index]


    fig, ax = plt.subplots()
    ax.errorbar(mu_values, mu_probabilities, color='black', ls='--')
    ax.axvline(x=mu_limit)
    plt.savefig('mu_pdf.pdf')
    plt.close()

    fig, ax = plt.subplots()
    ax.errorbar(mu_values, cumalitive_mu_PDF, color='black', ls='--')
    ax.hlines(0.05, mu_values[0], mu_values[-1], color='red', ls='dotted')
    ax.axvline(x=mu_limit)
    plt.savefig('mu_Cpdf.pdf')
    plt.close()
    print(mu_limit)


def get_single_mu_pval(data=None, couplings=None, mu_val=None):
    data_vals, data_errs = data['data']
    sig_vals, sig_errs = data['sig']
    bgd_vals, bgd_errs = data['bgd']

    def log_poisson(n,k):
        lp = -k
        lp[n!=0] += n[n!=0] + n[n!=0]*numpy.log(k[n!=0]/n[n!=0])
        return lp

    expectation_vals = lambda mu: mu * sig_vals + bgd_vals
    fixed_expectation_vals = expectation_vals(mu_val)
    loglikelihood = lambda n,mu: log_poisson(n,expectation_vals(mu)).sum()
    test_stat = lambda L,L_max,mu_max: -2*(L-L_max) if mu_max < mu_val else 0

    printa = lambda ar: print(' '.join([f'{a:.2f}' for a in ar])+'\n')

    def get_max_L_mu(n):
        mu_limit = 100
        for i in range(3):
            mu_array = numpy.linspace(-mu_limit,mu_limit,100)
            prior_L = -100
            prior_mu = 0
            for mu in mu_array:
                L = loglikelihood(n,mu)
                if L < prior_L:
                    mu_limit = mu*2
                    max_L = prior_L
                    max_mu = prior_mu
                    break
                prior_L = L
                prior_mu = mu
        if max_mu < 0: return loglikelihood(n,0), 0
        else: return max_L, max_mu


    num_toy_distros = 1000
    toy_test_stat_values = []
    for toy_index in range(num_toy_distros):
        toy_distro = numpy.random.poisson(fixed_expectation_vals)
        base_L = loglikelihood(toy_distro,mu_val)
        max_L, max_mu = get_max_L_mu(toy_distro)
        test_stat_val = test_stat(base_L, max_L, max_mu) 
        toy_test_stat_values.append(test_stat_val)
    #printa(toy_test_stat_values)
    pdf, bins = numpy.histogram(toy_test_stat_values, bins=100, range=(min(toy_test_stat_values),max(toy_test_stat_values)))
    pdf = pdf / pdf.sum()
    cumulative_pdf = pdf[::-1].cumsum()[::-1]
    observed_test_stat_value = test_stat(loglikelihood(data_vals,mu_val),*get_max_L_mu(data_vals))
    #print(observed_test_stat_value)
    observed_test_stat_bin = numpy.digitize(observed_test_stat_value, bins)
    p_value = cumulative_pdf[observed_test_stat_bin]
    return p_value


def make_mu_probability_distro(data=None, couplings=None):
    data_vals, data_errs = data['data']
    sig_vals, sig_errs = data['sig']
    bgd_vals, bgd_errs = data['bgd']

    mu_values = numpy.linspace(0,1000,10)
    p_values = []
    for mu in mu_values:
        pval = get_single_mu_pval(data=data, couplings=couplings, mu_val=mu)
        p_values.append(pval)
        print(pval)
        print()
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
    if len(sys.argv) > 1:
        # Load Signal - Juse use SM and then maybe cvv = 3
        sig_vals, sig_errs = load_signal(var_edges)

        # Load Data - Just load 3b1f Signal Region
        data_vals, data_errs = load_data(var_edges)

        # Load Background - Just reweight 2b Signal Region
        bgd_vals, bgd_errs = load_bgd(var_edges)
        #bgd_vals, bgd_errs = load_fake_bgd(data_vals, data_errs, sig_vals, sig_errs)

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

    make_data_display_plots(data=data,var_edges=var_edges)
    #make_lazy_mu_probability_distro(data=data, couplings=(1,1,1))
    make_mu_probability_distro(data=data, couplings=(1,1,1))
    #derive_maximum_likelihood_mu(data=data, couplings=(1,1,1))




    

main()
