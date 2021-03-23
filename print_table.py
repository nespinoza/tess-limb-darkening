import matplotlib.pyplot as plt
import numpy as np
import glob
import pickle

import juliet

def decimal_place(number, n):
    """
    This function returns the position of the n-th last decimals so you can print a result at n-decimal-places
    """
    str_number = format(number, '.20f')
    decimals = str_number.split('.')[-1]
    nonzero = False
    counter = 0
    for i in range(len(decimals)):
        if not nonzero:
            if decimals[i] != '0':
                nonzero = True
                counter = 1
        else:
            counter += 1
            if counter == n:
                break
    return i+1

def write_posteriors(planet, posteriors, fout, sectors):

    print(planet, sectors, posteriors.keys())
    # Extract b,p for 1-sector planets; convert sectors to string:
    if len(sectors) == 1:
        string_sectors = str(sectors[0])
    else:
        string_sectors = ','.join(np.array(sectors).astype('str'))

    if 'r1_p1' in posteriors.keys():
        b, p = juliet.reverse_bp(posteriors['r1_p1'], posteriors['r2_p1'], 0., 1.)
        posteriors['p_p1'], posteriors['b_p1'] = p, b

    posteriors['depth'] = (posteriors['p_p1']**2)*1e6
    posteriors['u1_TESS'], posteriors['u2_TESS'] = juliet.utils.reverse_ld_coeffs('quadratic', posteriors['q1_TESS'], posteriors['q2_TESS'])
    # Print posteriors:
    final_string = planet+'\t'
    for k in ['depth', 'p_p1', 'b_p1', 'q1_TESS', 'q2_TESS', 'u1_TESS', 'u2_TESS', 'a_p1', 't0_p1', 'P_p1']:

        val, valup, valdown = juliet.utils.get_quantiles(posteriors[k])
        err_up, err_down = valup-val, val-valdown

        if k != 'depth':
            # Check which of those errors is the smallest, to define the 2-decimal place string:
            min_val = np.min([err_up, err_down])

            # Extract number of decimal places to print:
            dp = decimal_place(min_val, 2)
            # String the values together for printing on the file:
            fmt_str = '{0:.'+str(dp)+'f}'+'\t'+'{1:.'+str(dp)+'f}'+'\t'+'{2:.'+str(dp)+'f}'
            # Stitch values together:
            final_string = final_string + fmt_str.format(val, err_up, err_down) + '\t'
        else:
            fmt_str = '{0:}'+'\t'+'{1:}'+'\t'+'{2:}'
            final_string = final_string + fmt_str.format(int(val+0.5), int(err_up+0.5), int(err_down+0.5)) + '\t'        

    # Save string to file:
    fout.write(final_string + string_sectors + '\n')

fout = open('results.dat', 'w')
folders = glob.glob('*b')
for folder in folders:
    # Load all datasets:
    datasets = glob.glob(folder+'/TESS*_in_transit')

    # First, check which sectors this object has data from:
    sectors = np.array([])
    for dataset in datasets:
        sector = int(dataset.split('/')[-1].split('_')[0].split('TESS')[1])
        if sector not in sectors:
            sectors = np.append(sectors, sector)

    # Order sectors numerically:
    idx = np.argsort(sectors)
    sectors = sectors[idx].astype(int)
    nsectors = len(sectors)

    # Now, for each sector, read posteriors for each model; check winner model, and take the samples from that model 
    # as the one for that sector. If no clear winner (lnZ <= 2), do model averaging: sample the same fraction of samples 
    # as the ratio between the posterior odds. For each of those, check the minimum number of samples (will be useful for 
    # weghing samples for model averaging):
    data = {}
    min_nsamples = np.inf
    epochs = np.zeros(len(sectors))
    counter = 0
    for sector in sectors:
        data[sector] = {}
        data[sector]['winner'] = {}
        data[sector]['ExpMatern'] = pickle.load(open(folder+'/TESS'+str(sector)+'_ExpMatern_in_transit/posteriors.pkl','rb'))
        data[sector]['QP'] = pickle.load(open(folder+'/TESS'+str(sector)+'_QP_in_transit/posteriors.pkl','rb'))
        if np.abs(data[sector]['ExpMatern']['lnZ'] - data[sector]['QP']['lnZ']) <= 2:
            # Calculate odds and total number of samples:
            odds = np.exp(data[sector]['ExpMatern']['lnZ'] - data[sector]['QP']['lnZ'])
            nsamples_expmatern, nsamples_qp = len(data[sector]['ExpMatern']['posterior_samples']['P_p1']), \
                                              len(data[sector]['QP']['posterior_samples']['P_p1'])
            max_nsamples = np.min([nsamples_expmatern, nsamples_qp])
            if odds > 1:
                idx = np.arange(nsamples_expmatern)
                idx_em = np.random.choice(idx,max_nsamples,replace=False)
                idx = np.arange(nsamples_qp)
                idx_qp = np.random.choice(idx,int(max_nsamples/odds), replace=False)
            else:
                idx = np.arange(nsamples_expmatern)
                idx_em = np.random.choice(idx,int(max_nsamples*odds),replace=False)
                idx = np.arange(nsamples_qp)
                idx_qp = np.random.choice(idx,max_nsamples, replace=False)

            for k in ['r1_p1', 'r2_p1', 'q1_TESS', 'q2_TESS', 'a_p1', 't0_p1', 'P_p1']:
                data[sector]['winner'][k] = np.append(data[sector]['ExpMatern']['posterior_samples'][k][idx_em],\
                                                      data[sector]['QP']['posterior_samples'][k][idx_qp])
        else:
            if data[sector]['ExpMatern']['lnZ'] > data[sector]['QP']['lnZ']:
                for k in ['r1_p1', 'r2_p1', 'q1_TESS', 'q2_TESS', 'a_p1', 't0_p1', 'P_p1']:
                    data[sector]['winner'][k] = data[sector]['ExpMatern']['posterior_samples'][k]
            else:
                for k in ['r1_p1', 'r2_p1', 'q1_TESS', 'q2_TESS', 'a_p1', 't0_p1', 'P_p1']:
                    data[sector]['winner'][k] = data[sector]['QP']['posterior_samples'][k]

        if counter == 0:
            tref, Pref = np.median(data[sector]['winner']['t0_p1']), np.median(data[sector]['winner']['P_p1'])
        else:
            t0, P = np.median(data[sector]['winner']['t0_p1']), np.median(data[sector]['winner']['P_p1'])
            epochs[counter] = int((t0 - tref)/P + 0.5)

        counter += 1
        c_samples = len(data[sector]['winner']['P_p1'])
        if c_samples < min_nsamples:
            min_nsamples = c_samples
        if nsectors == 1:
            write_posteriors(folder, data[sector]['winner'], fout, [sector])

    # Finally, get joint ("combined") fit from all sectors if available:
    if len(sectors)>1:
        sector = 'combined'
        data['combined'] = {}
        data['combined']['winner'] = {}
        if len(sectors) > 4:
            data['combined']['ExpMatern'] = pickle.load(open(folder+'/multisector_in_transit_ExpMatern/_dynesty_DNS_posteriors.pkl','rb'))
            data['combined']['QP'] = pickle.load(open(folder+'/multisector_in_transit_QP/_dynesty_DNS_posteriors.pkl','rb'))
        else:
            data['combined']['ExpMatern'] = pickle.load(open(folder+'/multisector_in_transit_ExpMatern/posteriors.pkl','rb'))
            data['combined']['QP'] = pickle.load(open(folder+'/multisector_in_transit_QP/posteriors.pkl','rb'))

        if np.abs(data['combined']['ExpMatern']['lnZ'] - data['combined']['QP']['lnZ']) <= 2:
            odds = np.exp(data[sector]['ExpMatern']['lnZ'] - data[sector]['QP']['lnZ'])
            nsamples_expmatern, nsamples_qp = len(data[sector]['ExpMatern']['posterior_samples']['P_p1']), \
                                              len(data[sector]['QP']['posterior_samples']['P_p1'])
            max_nsamples = np.min([nsamples_expmatern, nsamples_qp])
            if odds > 1:
                idx = np.arange(nsamples_expmatern)
                idx_em = np.random.choice(idx,max_nsamples,replace=False)
                idx = np.arange(nsamples_qp)
                idx_qp = np.random.choice(idx,int(max_nsamples/odds), replace=False)
            else:
                idx = np.arange(nsamples_expmatern)
                idx_em = np.random.choice(idx,int(max_nsamples*odds),replace=False)
                idx = np.arange(nsamples_qp)
                idx_qp = np.random.choice(idx,max_nsamples, replace=False)

            for k in data[sector]['ExpMatern']['posterior_samples'].keys():
                pname = k.split('_')[0]
                if pname in ['p', 'b', 'q1', 'q2', 'a', 't0', 'P']:
                    if pname[0] != 'q':
                        data[sector]['winner'][k] = np.append(data[sector]['ExpMatern']['posterior_samples'][k][idx_em],\
                                                          data[sector]['QP']['posterior_samples'][k][idx_qp])
                    else:
                        data[sector]['winner'][pname[0:2]+'_TESS'] = np.append(data[sector]['ExpMatern']['posterior_samples'][k][idx_em],\
                                                          data[sector]['QP']['posterior_samples'][k][idx_qp])
        else:
            if data[sector]['ExpMatern']['lnZ'] > data[sector]['QP']['lnZ']:
                for k in data[sector]['ExpMatern']['posterior_samples'].keys():
                    pname = k.split('_')[0]
                    if pname in ['p', 'b', 'q1', 'q2', 'a', 't0', 'P']:
                        if pname[0] != 'q':
                            data[sector]['winner'][k] = data[sector]['ExpMatern']['posterior_samples'][k]
                        else:
                            data[sector]['winner'][pname[0:2]+'_TESS'] = data[sector]['ExpMatern']['posterior_samples'][k]
            else:
                for k in data[sector]['QP']['posterior_samples'].keys():
                    pname = k.split('_')[0]
                    if pname in ['p', 'b', 'q1', 'q2', 'a', 't0', 'P']:
                        if pname[0] != 'q':
                            data[sector]['winner'][k] = data[sector]['QP']['posterior_samples'][k]
                        else:
                            data[sector]['winner'][pname[0:2]+'_TESS'] = data[sector]['QP']['posterior_samples'][k]

        write_posteriors(folder, data['combined']['winner'], fout, sectors)
        #for sector in sectors:
        #    plt.plot(data[sector]['winner']['q1_TESS'], data[sector]['winner']['q2_TESS'], '.', alpha = 0.1, color = 'black')
        #plt.plot(data['combined']['winner']['q1_TESS'], data['combined']['winner']['q2_TESS'], '.' , color='red',alpha=0.5)
fout.close()
#plt.xlabel('$q_1$')
#plt.ylabel('$q_2$')
#plt.show()
