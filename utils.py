import numpy as np
import juliet

def fit(t, f, ferr, sector, P, P_err, t0, t0_err, ecc, omega, GPmodel = 'ExpMatern', outpath = 'planetfit'):

    # Scale t0 to the transit closest to the center of the TESS observations:
    n = int((np.mean(t) - t0)/P)
    t0 += n*P
    t0_err = np.sqrt(t0_err**2 + (n * P_err)**2)

    # Port data in the juliet format:
    tt, ff, fferr = {}, {}, {}
    tt['TESS'], ff['TESS'], fferr['TESS'] = t, f, ferr

    # Define priors:
    priors = {}

    # First define parameter names, distributions and hyperparameters for GP-independant parameters:
    params1 = ['P_p1', 't0_p1', 'r1_p1', 'r2_p1', 'q1_TESS', 'q2_TESS', \
               'ecc_p1', 'omega_p1', 'a_p1', 'mdilution_TESS', 'mflux_TESS', 'sigma_w_TESS']

    dists1 = ['normal', 'normal', 'uniform', 'uniform', 'uniform', 'uniform', \
               'fixed','fixed','loguniform','fixed','normal','loguniform']

    hyperps1 = [[P,P_err], [t0, t0_err], [0., 1.], [0., 1.], [0., 1.], [0., 1.], \
               ecc, omega, [1., 100.], 1., [0., 0.1], [0.1, 10000.]]

    # Now define hyperparameters of the GP depending on the chosen kernel:
    if GPmodel == 'ExpMatern':
        params2 = ['GP_sigma_TESS', 'GP_timescale_TESS', 'GP_rho_TESS']
        dists2 = ['loguniform', 'loguniform', 'loguniform']
        hyperps2 = [[0.1, 10000.], [1e-3,1e2], [1e-3,1e2]]
    elif GPmodel == 'QP':
        params2 = ['GP_B_TESS', 'GP_C_TESS', 'GP_L_TESS', 'GP_Prot_TESS']
        dists2 = ['loguniform', 'loguniform', 'loguniform','loguniform']
        hyperps2 = [[1e-3,1e3], [1e-3,1e4], [1e-3, 1e2], [1.,1e2]]

    params = params1 + params2
    dists = dists1 + dists2
    hyperps = hyperps1 + hyperps2

    # Populate the priors dictionary:
    for param, dist, hyperp in zip(params, dists, hyperps):
        priors[param] = {}
        priors[param]['distribution'], priors[param]['hyperparameters'] = dist, hyperp

    # Run fit:
    dataset = juliet.load(priors=priors, t_lc = tt, y_lc = ff, \
                          yerr_lc = fferr, GP_regressors_lc = tt, out_folder = outpath+'_'+GPmodel)
    results = dataset.fit(n_live_points = 500)

def read_data(fname):
    fin = open(fname, 'r')
    data = {}
    while True:
        line = fin.readline()
        if line != '':
            if line[0] != '#':
                lv = line.split()
                name, ticid = lv[0], lv[1]
                data[name] = {}
                data[name]['ticid'] = ticid
        else:
            break
    return data
