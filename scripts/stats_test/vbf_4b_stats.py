#!/usr/bin/python
import sys
import math
import random
import statistics
import scipy
import scipy.stats
import sympy
import numpy
import pickle
from matplotlib import pyplot as plt
#import multiprocessing

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


def get_fast_cumulative_pval_function(bgd_yield, data_yield, mu_factor):
    base_couplings, basis_weights, basis_errors = fileioutils.load_signal_basis(None, pickle_load=True)
    basis_yields = [ sum(w) for w in basis_weights ]
    signal_formula = fileioutils.get_amplitude_function(base_couplings, form='expression')
    signal = signal_formula.subs([ (f'A{i}',y) for i,y in enumerate(basis_yields) ])
    sum_range = numpy.array(list(range(1,data_yield)))
    #sum_range = numpy.array([397,398,399,400])

    mu, s, b, n = sympy.symbols('u s b n')
    v = mu * s + b
    log_poisson_component = n*( sympy.log(v/n) - v/n + 1 )
    poisson = (2*sympy.pi*n)**(-1/2) * sympy.exp(log_poisson_component) # Unusable for n=0!
    poisson_expression = poisson.subs([(s,signal), (b, bgd_yield)])

    nax = numpy.newaxis
    poisson_function_sb = sympy.lambdify( [_k2v,_kl,_kv, 'n'], poisson_expression.subs('u',mu_factor), "numpy")
    poisson_function_b  = sympy.lambdify( ['n'], poisson_expression.subs([('u',0), (_k2v,0), (_kl,0), (_kv,0)]), "numpy")
    Cpoisson_function_sb = lambda kappas: poisson_function_sb(*kappas, sum_range[:,nax,nax]).sum(axis=0)
    Cpoisson_b = poisson_function_b(sum_range).sum(axis=0)
    Cpoisson_function_s = lambda kappas: Cpoisson_function_sb(kappas) / (1-Cpoisson_b)

    #print(Cpoisson_function_sb((*numpy.array([ [[2,3,4],[2,3,4]],[[3,3,3],[4,4,4]] ]), 1 )))
    #exit()
    return Cpoisson_function_s



def get_mu_kappa_expression(bgd_yield, data_yield):
    base_couplings, basis_weights, basis_errors = fileioutils.load_signal_basis(None, pickle_load=True)
    basis_yields = [ sum(w) for w in basis_weights ]
    signal_formula = fileioutils.get_amplitude_function(base_couplings, form='expression')
    signal = signal_formula.subs([ (f'A{i}',y) for i,y in enumerate(basis_yields) ])

    mu, s, b, n, N = sympy.symbols('u s b n N')
    v = mu * s + b
    log_poisson_component = n*( sympy.log(v/n) - v/n + 1 )
    poisson = (2*sympy.pi*n)**(-1/2) * sympy.exp(log_poisson_component) # Unusable for n=0!
    Cpoisson = sympy.concrete.summations.Sum(poisson, (n,1,N)) + sympy.exp(-v)
    Cpoisson_sb = Cpoisson
    Cpoisson_b = Cpoisson.subs(mu,0)
    Cpoisson_s = Cpoisson_sb / (1 - Cpoisson_b)
    mu_kappa_expression = Cpoisson_s.subs([(s,signal), (b, bgd_yield), (N, data_yield)])
    return mu_kappa_expression
    

def make_data_display_plots(results=None, var_edges=None, couplings=None, var_key='m_hh', var_title=r'$m_{HH}$'):
    fig, (ax, rat) = plt.subplots(2, gridspec_kw={'height_ratios':(2,1)}, sharex=True)
    ax.errorbar(var_edges[:-1]+0.5, results['data'][0], yerr=results['data'][1], marker='.', ls='--', color='purple', label='Data')
    ax.errorbar(var_edges[:-1]+0.5, results['bgd'][0], yerr=results['bgd'][1], marker='.', ls='--', color='blue', label='Bgd')
    ax.errorbar(var_edges[:-1]+0.5, results['sig'](couplings)[0], yerr=results['sig'](couplings)[1], marker='.', ls='--', color='green', label='Signal')

    sensitivity = results['sig'](couplings)[0] / numpy.sqrt( results['data'][1]**2 + results['bgd'][1]**2)
    rat.errorbar(var_edges[:-1]+0.5, sensitivity, marker='.', ls='--', color='red')
    rat.hlines(0, var_edges[0], var_edges[-1], linestyle='-', color='black')

    ax.legend()
    ax.set_ylabel('Yield of Events with Given '+var_title)
    rat.set_xlabel(var_title+' of Event')
    rat.set_ylabel('Significance')
    #plt.xlim(expectation*.5, expectation*1.5)
    #plt.ylim(0,1)
    coupling_title = _kappa_title + ' = ' + title_couplings(couplings)
    fig.suptitle(var_title+' Distribution'' for '+coupling_title)

    plt.savefig('out/'+'data_dump_'+var_key+'_'+name_couplings(couplings)+'.pdf')
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
    #exp_mus, exp_pvalues = get_mu_pvalue_relation((sig_yield, bgd_yield, int(bgd_yield)))
    #s1hi_mus, sigma1hi_pvalues = get_mu_pvalue_relation((sigmaV(sig_yield,1), sigmaV(bgd_yield,1), int(sigmaV(bgd_yield,1))))
    #s2hi_mus, sigma2hi_pvalues = get_mu_pvalue_relation((sigmaV(sig_yield,2), sigmaV(bgd_yield,2), int(sigmaV(bgd_yield,2))))
    #s1lo_mus, sigma1lo_pvalues = get_mu_pvalue_relation((sigmaV(sig_yield,-1), sigmaV(bgd_yield,-1), int(sigmaV(bgd_yield,-1))))
    exclusion_limit_index = numpy.argmax(pvalues < 0.05)
    mu_limit = mu_values[exclusion_limit_index]

    coupling_title = _kappa_title + ' = ' + title_couplings(couplings)
    fig, ax = plt.subplots()
    ax.plot(mu_values, pvalues, color='black', ls='-', label='Observed')
    #ax.plot(exp_mus, exp_pvalues, color='black', ls='--', label='Expected')
    #ax.plot(s1hi_mus, sigma1hi_pvalues, color='green', ls='--', label='$\pm 1 \sigma$')
    #ax.plot(s2hi_mus, sigma2hi_pvalues, color='yellow', ls='--', label='$\pm 1 \sigma$')
    #ax.plot(s1lo_mus, sigma1lo_pvalues, color='blue', ls='--', label='')
    ax.hlines(0.05, mu_values[0], mu_values[-1], color='red', ls='dotted', label='p-value = 0.05')
    ax.axvline(x=mu_limit, label=r'$\mu=$'f'{mu_limit:.2f}' )
    ax.legend()
    ax.grid()
    plt.ylabel('Signal p-value')
    plt.xlabel(r'Signal Scaling Coefficient $\mu$')
    plt.title(r'p-value vs $\mu$''\nfor '+coupling_title)
    plt.savefig('out/'+'mu_pvalue_'+name_couplings(couplings)+'.pdf')
    plt.close()

def mu_pval_scan(scan_coupling, coupling_list, plot_xvals,
        results, observed=True, slow_form=False, hl_lhc_projection=False):

    bgd_yield = sum(results['bgd'][0])
    data_yield = int(sum(results['data'][0])) if observed else int(bgd_yield)
    if hl_lhc_projection:
        hl_lhc_luminosity = 3000
        hl_lhc_scaling_factor = hl_lhc_luminosity / 126.7
        bgd_yield *= hl_lhc_scaling_factor
        data_yield = int(bgd_yield)
        bgd_yield

    mu_limit_list = []
    if slow_form:
        for couplings in coupling_list:
            sig_yield = sum(results['sig'](couplings)[0])
            print(f'Deriving: {couplings}...', end='')
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
            print(f'Deriving: {x:.2f}...', end='')
            if scan_coupling == 'k2v':
                mu_expression = mu_kappa_expression.subs([(_k2v,x),(_kl,1),(_kv,1)]) - 0.05
            elif scan_coupling == 'kl':
                mu_expression = mu_kappa_expression.subs([(_k2v,1),(_kl,x),(_kv,1)]) - 0.05
            if hl_lhc_projection: mu_expression = mu_expression.subs('u', f'{hl_lhc_scaling_factor}*u')
            mu_function = sympy.lambdify( ['u'], mu_expression, "numpy")
            mu_vals = numpy.linspace(0,100,1001)
            rough_mu = mu_function(mu_vals)
            mu_guess = mu_vals[numpy.argmax(rough_mu < 0)]
            mu_limit = scipy.optimize.fsolve(mu_function, mu_guess)[0]
            mu_limit_list.append(mu_limit)
            print(f'mu = {mu_limit:.2f}')
    mu_limit_array = numpy.array(mu_limit_list)
    return mu_limit_array


def make_basic_1D_mu_plot(results=None, scan_coupling=None, slow_form=False, hl_lhc_projection=False):
    if scan_coupling == 'k2v':
        plot_xvals = numpy.linspace(-2,4,13)
        if not slow_form: plot_xvals = numpy.linspace(-2,4,6*10+1)
        coupling_list = [ (k2v,1,1) for k2v in plot_xvals ]
    elif scan_coupling == 'kl':
        plot_xvals = numpy.linspace(-20,20,21)
        if not slow_form: plot_xvals = numpy.linspace(-20,20,40*10+1)
        coupling_list = [ (1,kl,1) for kl in plot_xvals ]
    else: return

    xsec_fn = fileioutils.get_theory_combination_function()
    theory_xsec = numpy.array([ xsec_fn(c) for c in coupling_list ])

    mu_limit_array = mu_pval_scan(scan_coupling, coupling_list, plot_xvals,
        results, slow_form=slow_form, hl_lhc_projection=hl_lhc_projection)

    #exp_mu_limit_array = mu_pval_scan(scan_coupling, coupling_list, plot_xvals,
        #results, observed=False, slow_form=slow_form)

    infix = 'slow_' if slow_form else 'fast_'
    if hl_lhc_projection: infix += 'HL-LHC_'
    fig, ax = plt.subplots()
    if not hl_lhc_projection:
        ax.plot(plot_xvals, mu_limit_array, color='black', ls='-', label='Observed Limit')
    else:
        ax.plot(plot_xvals, mu_limit_array, color='black', ls='--', label='Expected Limit')
    #ax.plot(plot_xvals, exp_mu_limit_array, color='black', ls='--', label='Expected Limit')
    ax.set_yscale('log')
    ax.axhline(1, color='red', ls='-', label=r'$\mu=1$')
    ax.legend()
    ax.grid()
    plt.xlabel(_coupling_labels[scan_coupling])
    plt.ylabel(r'$\mu$ Value Required for 95% Confidence')
    plt.title(_coupling_labels[scan_coupling] + r' $\mu$ Limit Scan')
    plt.savefig('out/'+'mu_limits_'+infix+scan_coupling+'.pdf')
    plt.close()

    fig, ax = plt.subplots()
    if not hl_lhc_projection:
        ax.plot(plot_xvals, mu_limit_array*theory_xsec, color='black', ls='-', label='Observed Limit')
    else:
        ax.plot(plot_xvals, mu_limit_array*theory_xsec, color='black', ls='--', label='Expected Limit')
    #ax.plot(plot_xvals, exp_mu_limit_array*theory_xsec, color='black', ls='--', label='Expected Limit')
    ax.set_yscale('log')
    ax.plot(plot_xvals, theory_xsec, color='red', ls='-', label='Theory cross-section')
    ax.set_ylim(1,1e4)
    ax.grid()
    plt.xlabel(_coupling_labels[scan_coupling])
    plt.ylabel(r'Cross-section at 95% Confidence')
    plt.title(_coupling_labels[scan_coupling] + r' Cross-section Limit Scan')
    ax.legend()
    plt.savefig('out/'+'xsec_limits_'+infix+scan_coupling+'.pdf')
    plt.close()


def plot_slice(kappas, coupling_parameters, Cpoisson_function_s, shell_points, infix, si, key, kslice):
    limit_resolution = 100

    print('    '+key+' = '+str(kslice))
    (xi, xkey), (yi, ykey) = [ (i,k) for i,k in enumerate(kappas) if k != key ]
    xbounds, _ = coupling_parameters[xkey]
    ybounds, _ = coupling_parameters[ykey]

    xrange = numpy.linspace(*xbounds,limit_resolution)
    yrange = numpy.linspace(*ybounds,limit_resolution)
    xy_grid = numpy.meshgrid(xrange, yrange)
    kappa_grid = [None, None, None]
    kappa_grid[si] = kslice
    kappa_grid[xi] = xy_grid[0]
    kappa_grid[yi] = xy_grid[1]
    import warnings
    warnings.filterwarnings('ignore')
    pvalue_grid = Cpoisson_function_s(kappa_grid)
    #pvalue_grid = kappa_function(*kappa_grid)
    #pvalue_exp_grid = kappa_exp_function(*kappa_grid)

    fig, ax = plt.subplots()
    contour_group = ax.contour(xy_grid[0], xy_grid[1], pvalue_grid, levels=[0.05], antialiased=True)
    #ax.contour(xy_grid[0], xy_grid[1], pvalue_exp_grid, levels=[0.05], antialiased=True)

    for path in contour_group.collections[0].get_paths():
        for x, y in path.vertices:
            new_point = [0,0,0]
            new_point[si] = kslice
            new_point[xi] = x
            new_point[yi] = y
            shell_points.append(new_point)

    plt.grid()
    plt.title('Limit Boundaries for ' +_coupling_labels[key]+ f' = {kslice:.2f}')
    plt.xlabel(_coupling_labels[xkey])
    plt.ylabel(_coupling_labels[ykey])
    plt.savefig('out/3D/limit_slice_'+infix+key+'_'+str(kslice).replace('.','p')+'.pdf')
    plt.close()



def make_multidimensional_limit_plots(results=None, hl_lhc_projection=False):
    #k2v_bounds = (-4,15)
    #kl_bounds  = (-50,50)
    #kv_bounds  = (-6,6)
    k2v_bounds = (-2.5,14)
    kl_bounds  = (-51,49)
    kv_bounds  = (-3.5,3.5)

    k2v_slices  = numpy.linspace(*k2v_bounds, int((k2v_bounds[1]-k2v_bounds[0])*2)+1)
    #kl_slices  = [-50, -40, -30, -20, -10, 0, 1, 10, 20, 30, 40, 50]
    kl_slices  = numpy.linspace(*kl_bounds, 26)
    kv_slices  = numpy.linspace(*kv_bounds, int((kv_bounds[1]-kv_bounds[0])*2)+1)
    #k2v_slices = numpy.linspace(*k2v_bounds, 11)
    #kl_slices  = [-50, -30, -10, 0, 10, 30, 50]
    #kv_slices  = numpy.linspace(*kv_bounds, 7)

    coupling_parameters = {
        'k2v': [k2v_bounds, k2v_slices],
        'kl':  [kl_bounds, kl_slices],
        'kv':  [kv_bounds, kv_slices]
    }


    bgd_yield = sum(results['bgd'][0])
    data_yield = int(sum(results['data'][0]))
    mu_factor = 1
    infix = ''
    if hl_lhc_projection:
        infix = 'HL-LHC_'
        hl_lhc_luminosity = 3000
        hl_lhc_scaling_factor = hl_lhc_luminosity / 126.7
        bgd_yield *= hl_lhc_scaling_factor
        data_yield = int(bgd_yield)
        mu_factor = hl_lhc_scaling_factor
    #kappa_exp_expression = get_poisson_expression( bgd_yield, int(bgd_yield)).subs('u',1)
    #kappa_exp_function = sympy.lambdify( [_k2v,_kl,_kv], kappa_expression, "numpy")

    Cpoisson_function_s = get_fast_cumulative_pval_function(bgd_yield, data_yield, mu_factor)
    #kappa_expression = get_mu_kappa_expression(bgd_yield, data_yield).subs('u',mu_factor)
    #kappa_function = sympy.lambdify( [_k2v,_kl,_kv], kappa_expression, "numpy")
    #Cpoisson_function_s = lambda kappas: kappa_function(*kappas)

    kappas = list(coupling_parameters)
    print('Generating multi-dimensional pvalues...')
    shell_points = []
    #process_list = []
    #max_processes = 8
    for si, key in enumerate(kappas):
        sbounds, srange = coupling_parameters[key]
        for kslice in srange:
            #if key != 'kl' or kslice != 1: continue
            plot_slice(kappas, coupling_parameters, Cpoisson_function_s, shell_points, infix, si, key, kslice) 
            #if len(process_list) >= max_processes:
            #    for process in process_list: process.join()
            #    process_list = []
            #    
            #args=(kappas, coupling_parameters, Cpoisson_function_s, shell_points, infix, si, key, kslice) 
            #slice_process = multiprocessing.Process(target=plot_slice, args=args)
            #process_list.append(slice_process)
            #slice_process.start()
    #for process in process_list: process.join()

    return shell_points
    

def make_full_3D_render(shell_points):
    #numpy.savetxt('mesh_dump.dat', numpy.array(shell_points).transpose())
    numpy.savetxt('mesh_dump.dat', numpy.array(shell_points))
    #shell_points = [ [x,y,z] for x,y,z in zip(*shell_points) ]
    shell_points.sort(key=lambda p: p[2])
    shell_points = numpy.array(shell_points)

    import inspect
    import pymesh
    print('\n'.join([str(i) for i in inspect.getmembers(pymesh)]))
    tri_form = pymesh.triangle()
    tri_form.points = shell_points
    tri_form.run()
    mesh = tri_form.mesh
    mesh.save_mesh('limit_mesh.stl')



    #from stl import mesh
    #triangulation = None
    #resolution = 5
    #for zsplit in numpy.array_split(shell_points, 5):
    #    vertices = zsplit
    #    if triangulation is None:
    #        triangulation = scipy.spatial.Delaunay(vertices, incremental=True)
    #    else:
    #        triangulation.add_points(vertices, restart=False)
    #triangulation.close()

    #vertices = triangulation.points
    ##triangulation = scipy.spatial.Delaunay(vertices)
    #faces = triangulation.convex_hull
    #limit_mesh = mesh.Mesh(numpy.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))
    #for i, f in enumerate(faces):
    #    for j in range(3):
    #            limit_mesh.vectors[i][j] = vertices[f[j],:]
    ##limit_mesh.save(f'limit_mesh_{si}.stl')
    #limit_mesh.save(f'limit_mesh.stl')




def main():
    pickle_load = len(sys.argv) > 1
    #var_edges = numpy.linspace(200, 1400, 30)
    #var_key = 'm_hh'
    var_edges = numpy.linspace(0, 4, 20)
    var_key = 'dEta_hh'

    ###################
    # LOAD EVERYTHING #
    ###################
    # Load Signal
    signal = fileioutils.load_signal(var_edges, pickle_load=pickle_load, var_key=var_key)

    # Load ggF Background
    #ggfB_vals, ggfB_errs = fileioutils.load_ggF_bgd(var_edges)

    # Load Data
    data_vals, data_errs = fileioutils.load_data(var_edges, pickle_load=pickle_load, var_key=var_key)

    # Load Background
    bgd_vals, bgd_errs = fileioutils.load_bgd(var_edges, pickle_load=pickle_load, var_key=var_key)

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

    #make_lazy_mu_probability_distro(results=results, couplings=(1,1,1))
    #make_lazy_mu_probability_distro(results=results, couplings=(3,1,1))

    #make_basic_1D_mu_plot(results=results, scan_coupling='k2v', slow_form=False)
    #make_basic_1D_mu_plot(results=results, scan_coupling='kl', slow_form=False)

    #shell_points = make_multidimensional_limit_plots(results=results)
    #pickle.dump(shell_points, open('.shell_points.p','wb'))
    #shell_points = pickle.load(open('.shell_points.p','rb'))
    #make_full_3D_render(shell_points)

    make_basic_1D_mu_plot(results=results, scan_coupling='k2v', slow_form=False, hl_lhc_projection=True)
    make_basic_1D_mu_plot(results=results, scan_coupling='kl', slow_form=False, hl_lhc_projection=True)
    #shell_points = make_multidimensional_limit_plots(results=results, hl_lhc_projection=True)

    #make_data_display_plots(results=results, var_key=var_key, var_edges=var_edges, couplings=(-1,1,1))
    #make_data_display_plots(results=results, var_key=var_key, var_edges=var_edges, couplings=(1,1,1))
    #make_data_display_plots(results=results, var_key=var_key, var_edges=var_edges, couplings=(2,1,1))
    #make_data_display_plots(results=results, var_key=var_key, var_edges=var_edges, couplings=(3,1,1))
    #make_data_display_plots(results=results, var_key=var_key, var_edges=var_edges, couplings=(1,10,1))


main()
