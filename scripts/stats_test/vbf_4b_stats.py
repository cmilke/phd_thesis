#!/usr/bin/python
import sys
import math
import random
import statistics
import scipy
import scipy.stats
import sympy
import numpy
from matplotlib import pyplot as plt

import fileioutils
from fileioutils import _k2v, _kl, _kv

_coupling_labels = {
    'k2v': r'$\kappa_{2V}$',
    'kl':  r'$\kappa_{\lambda}$',
    'kv':  r'$\kappa_{V}$'
}

_kappa_title = ', '.join(_coupling_labels.values())


def name_couplings(couplings):
    return '_'.join([f'{c:.2f}' for c in couplings]).replace('.','p')

def title_couplings(couplings):
    return ', '.join([f'{c}' for c in couplings])

def log_poisson(n,v):
    lp = numpy.full(len(n),-v).astype(float)
    nz = n!=0
    n = n[nz]
    lp[nz] = n - v + n*numpy.log(v/n) - (1/2)*numpy.log(2*math.pi*n)
    return lp


def get_mu_kappa_expression(bgd_yield, data_yield):
    base_couplings, basis_weights, basis_errors = fileioutils.load_signal_basis(None, pickle_load=True)
    basis_yields = [ sum(w) for w in basis_weights ]
    signal_formula = fileioutils.get_amplitude_function(base_couplings, form='expression')
    signal = signal_formula.subs([ (f'A{i}',y) for i,y in enumerate(basis_yields) ])

    mu, s, b, n, N = sympy.symbols('u s b n N')
    v = mu * s + b
    #poisson = v**n * sympy.exp(-v) / sympy.factorial(n)
    poisson = (2*sympy.pi*n)**(-1/2) * (v/n)**n * sympy.exp(n-v) # Unusable for n=0!
    Cpoisson = sympy.concrete.summations.Sum(poisson, (n,1,N)) + sympy.exp(-v)
    Cpoisson_sb = Cpoisson
    Cpoisson_b = Cpoisson.subs(mu,0)
    Cpoisson_s = Cpoisson_sb / (1 - Cpoisson_b)
    mu_kappa_expression = Cpoisson_s.subs([(s,signal), (b, bgd_yield), (N, data_yield)])
    return mu_kappa_expression
    

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


    plt.savefig('out/'+'data_dump_'+name_couplings(couplings)+'.pdf')
    plt.close()



def make_sb_poisson_plots(results=None, prefix='', couplings=None):
    mu = 1
    signal = mu*sum(results['sig'](couplings)[0])
    background = int(sum(results['bgd'][0]))
    observed = int(sum(results['data'][0]))

    expectation = signal+background
    max_n = int(expectation)*4 # This is basically just the range of the plots
    poisson_inputs = numpy.arange(0,max_n,1)
    log_poisson_bgd_values = log_poisson(poisson_inputs,background)
    log_poisson_bs_values = log_poisson(poisson_inputs,expectation)
    poisson_bgd_values = numpy.exp(log_poisson_bgd_values)
    poisson_bs_values = numpy.exp(log_poisson_bs_values)
    poisson_bgd_values = poisson_bgd_values / poisson_bgd_values.sum()
    poisson_bs_values = poisson_bs_values / poisson_bs_values.sum()
    #cumulative_bgd_poisson = poisson_bgd_values[::-1].cumsum()[::-1]
    #cumulative_bs_poisson = poisson_bs_values[::-1].cumsum()[::-1]
    cumulative_bgd_poisson = poisson_bgd_values.cumsum()
    cumulative_bs_poisson = poisson_bs_values.cumsum()
    cumulative_sig_poisson = cumulative_bs_poisson / (1 - cumulative_bgd_poisson)
    sig_pvalue = cumulative_sig_poisson[observed]
    bgd_pvalue = cumulative_bgd_poisson[observed]
    bs_pvalue = cumulative_bs_poisson[observed]

    coupling_title = _kappa_title + ' = ' + title_couplings(couplings)

    fig, ax = plt.subplots()
    ax.plot(poisson_inputs, poisson_bgd_values, label='Background PDF', color='blue')
    ax.plot(poisson_inputs, poisson_bs_values, ls='--', label='S+B PDF', color='green')
    ax.axvline(observed, ls='--', label=f'Observed n={observed}', color='red')
    ax.fill_between(range(0,observed+1), 0, poisson_bgd_values[:observed+1], color='blue', hatch='///', alpha=0.3, label=f'bgd p-value={bgd_pvalue:.2f}')
    ax.fill_between(range(0,observed+1), 0, poisson_bs_values[:observed+1], color='green', hatch='/', alpha=0.3, label=f'S+B p-value={bs_pvalue:.2f}')
    plt.xlabel('Number of Events')
    plt.ylabel('Poisson Probability')
    plt.xlim(expectation*.5, expectation*1.5)
    plt.ylim(0)
    ax.legend()
    plt.title('Poisson Distributions\nfor '+coupling_title)
    plt.savefig('out/'+prefix+'_poisson_'+name_couplings(couplings)+'.pdf')
    plt.close()

    fig, ax = plt.subplots()
    ax.plot(poisson_inputs, cumulative_bgd_poisson, label='Bgd Cumulative PDF')
    ax.plot(poisson_inputs, cumulative_bs_poisson, label='S+B Cumulative PDF', ls='--')
    ax.plot(poisson_inputs, cumulative_sig_poisson, label='Signal Cumulative PDF', color='purple')
    ax.axvline(observed, ls='--', label=f'Observed n={observed}', color='red')
    ax.axhline(bgd_pvalue, ls='dotted', label=f'Bgd p-value={bgd_pvalue:.2f}', color='blue')
    ax.axhline(sig_pvalue, ls='dotted', label=f'Signal p-value={sig_pvalue:.2f}', color='purple')
    plt.xlabel('Number of Events')
    plt.ylabel('Cumulative Poisson Probability')
    plt.xlim(expectation*.5, expectation*1.5)
    plt.ylim(0,1)
    ax.legend(fontsize=8)
    plt.title('Cumulative Poisson Distributions\nfor '+coupling_title)
    plt.savefig('out/'+prefix+'_Cpoisson_'+name_couplings(couplings)+'.pdf')
    plt.close()


def get_mu_pvalue_relation(yields, max_mu=1000, zoom_passes=2, precision_threshold=1e-5):
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
        #cumulative_bgd_poisson = poisson_bgd_values[::-1].cumsum()[::-1]
        #cumulative_bs_poisson = poisson_bs_values[::-1].cumsum()[::-1]
        cumulative_bgd_poisson = poisson_bgd_values.cumsum()
        cumulative_bs_poisson = poisson_bs_values.cumsum()
        cumulative_sig_poisson = cumulative_bs_poisson / (1 - cumulative_bgd_poisson)
        sig_pvalue = cumulative_sig_poisson[observed]
        bgd_pvalue = cumulative_bgd_poisson[observed]
        bs_pvalue = cumulative_bs_poisson[observed]
        return sig_pvalue

    sig_yield, bgd_yield, data_yield = yields

    for i in range(zoom_passes+1):
        mu_values = numpy.linspace(0,max_mu,100)
        pvalues = [ get_pvalue(mu*sig_yield, bgd_yield, data_yield) for mu in mu_values ]
        pvalues = numpy.array(pvalues)
        max_mu = mu_values[numpy.argmax(pvalues<precision_threshold)]
    return mu_values, pvalues


def make_lazy_mu_probability_distro(results=None, couplings=None):
    data_yield = int(sum(results['data'][0]))
    sig_yield = sum(results['sig'](couplings)[0])
    bgd_yield = sum(results['bgd'][0])

    sigmaV = lambda y,hilo: y+hilo*math.sqrt(y)
    mu_values, pvalues = get_mu_pvalue_relation((sig_yield, bgd_yield, data_yield))
    exp_mus, exp_pvalues = get_mu_pvalue_relation((sig_yield, bgd_yield, int(bgd_yield)))
    #s1hi_mus, sigma1hi_pvalues = get_mu_pvalue_relation((sigmaV(sig_yield,1), sigmaV(bgd_yield,1), int(sigmaV(bgd_yield,1))))
    #s2hi_mus, sigma2hi_pvalues = get_mu_pvalue_relation((sigmaV(sig_yield,2), sigmaV(bgd_yield,2), int(sigmaV(bgd_yield,2))))
    #s1lo_mus, sigma1lo_pvalues = get_mu_pvalue_relation((sigmaV(sig_yield,-1), sigmaV(bgd_yield,-1), int(sigmaV(bgd_yield,-1))))
    exclusion_limit_index = numpy.argmax(pvalues < 0.05)
    mu_limit = mu_values[exclusion_limit_index]

    coupling_title = _kappa_title + ' = ' + title_couplings(couplings)
    fig, ax = plt.subplots()
    ax.plot(mu_values, pvalues, color='black', ls='-', label='Observed')
    ax.plot(exp_mus, exp_pvalues, color='black', ls='--', label='Expected')
    #ax.plot(s1hi_mus, sigma1hi_pvalues, color='green', ls='--', label='$\pm 1 \sigma$')
    #ax.plot(s2hi_mus, sigma2hi_pvalues, color='yellow', ls='--', label='$\pm 1 \sigma$')
    #ax.plot(s1lo_mus, sigma1lo_pvalues, color='blue', ls='--', label='')
    ax.hlines(0.05, mu_values[0], mu_values[-1], color='red', ls='dotted', label='p-value = 0.05')
    ax.axvline(x=mu_limit, label=r'$\mu=$'f'{mu_limit:.2f}' )
    ax.legend()
    plt.ylabel('Signal p-value')
    plt.xlabel(r'Signal Scaling Coefficient $\mu$')
    plt.title(r'p-value vs $\mu$''\nfor '+coupling_title)
    plt.savefig('out/'+'mu_pvalue_'+name_couplings(couplings)+'.pdf')
    plt.close()


def make_basic_1D_mu_plot(results=None, scan_coupling=None, slow_form=False):
    data_yield = int(sum(results['data'][0]))
    bgd_yield = sum(results['bgd'][0])

    if scan_coupling == 'k2v':
        plot_xvals = numpy.linspace(-2,4,13)
        if not slow_form: plot_xvals = numpy.linspace(-2,4,6*10+1)
        coupling_list = [ (k2v,1,1) for k2v in plot_xvals ]
    else: return

    xsec_fn = fileioutils.get_theory_combination_function()
    theory_xsec = numpy.array([ xsec_fn(c) for c in coupling_list ])

    mu_limit_list = []
    if slow_form:
        for couplings in coupling_list:
            print(f'Deriving: {couplings}...', end='')
            sig_yield = sum(results['sig'](couplings)[0])
            mu_values, pvalues = get_mu_pvalue_relation((sig_yield, bgd_yield, data_yield))
            exclusion_limit_index = numpy.argmax(pvalues < 0.05)
            mu_limit = mu_values[exclusion_limit_index]
            mu_limit_list.append(mu_limit)
            print(f'mu = {mu_limit:.2f}')
    else:
        mu_kappa_expression = get_mu_kappa_expression(bgd_yield, data_yield)
        import warnings
        warnings.filterwarnings('ignore')
        for x in plot_xvals:
            print(f'Deriving: {x}...', end='')
            if scan_coupling == 'k2v':
                mu_expression = mu_kappa_expression.subs([(_k2v,x),(_kl,1),(_kv,1)]) - 0.05
            mu_function = sympy.lambdify( ['u'], mu_expression, "numpy")
            mu_vals = numpy.linspace(0,100,1001)
            rough_mu = mu_function(mu_vals)
            mu_guess = mu_vals[numpy.argmax(rough_mu < 0)]
            mu_limit = scipy.optimize.fsolve(mu_function, mu_guess)[0]
            mu_limit_list.append(mu_limit)
            print(f'mu = {mu_limit:.2f}')
    mu_limit_array = numpy.array(mu_limit_list)
    for k2v, pval in zip(plot_xvals, mu_limit_array):
        print(f'{k2v:.2f}, {pval:.2f}')

    slow = 'slow_' if slow_form else 'fast_'
    fig, ax = plt.subplots()
    ax.plot(plot_xvals, mu_limit_array, color='black', ls='-', label='Observed Limit')
    ax.set_yscale('log')
    ax.axhline(1, color='red', ls='-', label=r'$\mu=1$')
    ax.legend()
    plt.savefig('out/'+'mu_limits_'+slow+scan_coupling+'.pdf')
    plt.close()

    fig, ax = plt.subplots()
    ax.plot(plot_xvals, mu_limit_array*theory_xsec, color='black', ls='-', label='Observed Limit')
    ax.set_yscale('log')
    ax.plot(plot_xvals, theory_xsec, color='red', ls='-', label='Theory cross-section')
    ax.set_ylim(1,1e4)
    ax.legend()
    plt.savefig('out/'+'xsec_limits_'+slow+scan_coupling+'.pdf')
    plt.close()



def make_multidimensional_limit_plots(results=None):
    k2v_bounds = (-4,15)
    kl_bounds  = (-50,50)
    kv_bounds  = (-6,6)

    k2v_slices = numpy.linspace(*k2v_bounds, 20)
    kl_slices  = [-50, -40, -30, -20, -10, 0, 1, 10, 20, 30, 40, 50]
    kv_slices  = numpy.linspace(*kv_bounds, 13)

    coupling_parameters = {
        'k2v': [k2v_bounds, k2v_slices, 'k2v'],
        'kl':  [kl_bounds, kl_slices, 'kl'],
        'kv':  [kv_bounds, kv_slices, 'kv']
    }
    limit_resolution = 100


    data_yield = int(sum(results['data'][0]))
    bgd_yield = sum(results['bgd'][0])
    kappa_expression = get_mu_kappa_expression( bgd_yield, data_yield).subs('u',1)
    kappa_function = sympy.lambdify( [_k2v,_kl,_kv], kappa_expression, "numpy")


    kappas = list(coupling_parameters)
    print('Generating multi-dimensional pvalues...')
    for si, key in enumerate(kappas):
        sbounds, srange, stitle = coupling_parameters[key]
        for kslice in srange:
            print('    '+key+' = '+str(kslice))
            (xi, xkey), (yi, ykey) = [ (i,k) for i,k in enumerate(kappas) if k != key ]
            xbounds, _, xtitle = coupling_parameters[xkey]
            ybounds, _, ytitle = coupling_parameters[ykey]

            xrange = numpy.linspace(*xbounds,limit_resolution)
            yrange = numpy.linspace(*ybounds,limit_resolution)
            xy_grid = numpy.meshgrid(xrange, yrange)
            kappa_grid = [None, None, None]
            kappa_grid[si] = kslice
            kappa_grid[xi] = xy_grid[0]
            kappa_grid[yi] = xy_grid[1]
            #print(kappa_function(*kappa_matrix))
            import warnings
            warnings.filterwarnings('ignore')
            pvalue_grid = kappa_function(*kappa_grid)

            fig, ax = plt.subplots()
            ax.contour(xy_grid[0], xy_grid[1], pvalue_grid, levels=[0.05], antialiased=True)
            plt.grid()
            plt.title(stitle)
            plt.xlabel(xtitle)
            plt.ylabel(ytitle)
            plt.savefig('out/3D/limit_slice_'+key+'_'+str(kslice).replace('.','p')+'.pdf')
            plt.close()
    






def main():
    pickle_load = len(sys.argv) > 1
    var_edges = numpy.linspace(200, 1400, 30)

    ###################
    # LOAD EVERYTHING #
    ###################
    # Load Signal
    signal = fileioutils.load_signal(var_edges, pickle_load=pickle_load)

    # Load ggF Background
    #ggfB_vals, ggfB_errs = fileioutils.load_ggF_bgd(var_edges)

    # Load Data
    data_vals, data_errs = fileioutils.load_data(var_edges, pickle_load=pickle_load)

    # Load Background
    bgd_vals, bgd_errs = fileioutils.load_bgd(var_edges, pickle_load=pickle_load)

    results = {
        'data': (data_vals, data_errs),
        'bgd' : (bgd_vals, bgd_errs),
        'sig' : signal,
        #'ggfB' : (ggfB_vals, ggfB_errs)
    }
    #pickle.dump(results, open(cache_file,'wb'))
    #results = pickle.load(open(cache_file,'rb'))

    print('Data Loaded')

    #print('Signal')
    #print(signal((3,1,1))[0].sum(), signal((3,1,1))[1].sum())
    #print()
    ##print('ggF Bgd')
    ##print(ggfB_vals.sum(), ggfB_vals.sum())
    ##print()
    #print('Bgd')
    #print(results['bgd'][0].sum(), results['bgd'][1].sum())
    #print()
    #print('Data')
    #print(results['data'][0].sum(), results['data'][1].sum())

    #make_sb_poisson_plots(results=results, prefix='total_yield', couplings=(1,1,1))
    #make_sb_poisson_plots(results=results, prefix='total_yield', couplings=(3,1,1))
    make_lazy_mu_probability_distro(results=results, couplings=(1,1,1))
    make_lazy_mu_probability_distro(results=results, couplings=(3,1,1))
    #make_basic_1D_mu_plot(results=results, scan_coupling='k2v', slow_form=False)
    #make_multidimensional_limit_plots(results=results)
    #make_data_display_plots(results=results,var_edges=var_edges)


main()
