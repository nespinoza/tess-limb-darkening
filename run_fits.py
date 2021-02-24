import numpy as np
import juliet
import utils

# Exposure time of 2-min cadence data, in days:
exp_time = (2./60.)/24.

# Load target list. This will only load the planet name and TIC IDs:
target_list = utils.read_data('data.dat')

# Iterate through all planets:
for planet in target_list.keys():

    print('Working on ',planet)
    # Load all data for this particular planet; if data exists, go forward. If not, skip 
    # target:
    try:

        planet_data, url = utils.get_target_data(planet)

        # Extract useful data; if not floats, ommit target:
        tdur = planet_data['transit_duration']
        tdepth = planet_data['transit_depth']
        period = planet_data['orbital_period']
        period_err = (planet_data['orbital_period_upper'] + planet_data['orbital_period_lower'])*0.5
        t0 = planet_data['transit_time'] + 2400000.5
        t0_err = (planet_data['transit_time_upper'] + planet_data['transit_time_lower'])*0.5

        if (type(tdur) is float) and (type(tdepth) is float) and (type(period) is float) and (type(period_err) is float) and \
           (type(t0) is float) and (type(t0_err) is float):
            has_data = True
        else:
            has_data = False

        # Now check eccentricity and omega. If no data, set to 0 and 90:
        ecc, omega = planet_data['eccentricity'], planet_data['omega']
        if (type(ecc) is not float):
            ecc = 0.
            omega = 90.

    except:
        print('No planetary data for ',planet)
        has_data = False

    # If it has data, we move ahead with the analysis:
    if has_data:

        # First, load data for each sector:
        t, f, ferr = juliet.utils.get_all_TESS_data('TIC ' + target_list['WASP-6']['ticid'])

        # If it has planetary data, go through each sector. First, estimate the transit depth precision 
        # (assuming a box-shaped transit) we would obtain with this 2-min TESS data. If this gives rise 
        # to a 5-sigma "detection" of the depth using all the (phased) transits, we analyze it:
        for sector in t.keys():

            # Estimate number of transits we should expect in this dataset:
            total_time = np.max(t[sector]) - np.min(t[sector])
            Ntransits = int(total_time/period)

            # Estimate number of datapoints in-transit in the phased lightcurve:
            Nin = int(tdur/exp_time) * Ntransits

            # Estimate transit depth precision:
            I = 2. * Nin
            sigma = np.median(ferr[sector])
            sigma_depth = (2. * sigma)/np.sqrt(I)

            # If larger than 5-sigma, perform the fit:
            if tdepth/sigma_depth > 5:
