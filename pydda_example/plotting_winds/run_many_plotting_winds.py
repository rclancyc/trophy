import os

import pyart
import sys
import time
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import warnings
warnings.filterwarnings("ignore")

# change to location of trophy repo
trophy_repo_path = os.environ['GITREPOS'] + '/trophy/'
sys.path.insert(0, trophy_repo_path + 'pydda_example/')
sys.path.insert(0, trophy_repo_path + 'pydda_example/dynTRpydda_edits/')

import dynTRpydda_edits as pydda

berr_grid = pyart.io.read_grid(pydda.tests.EXAMPLE_RADAR0)
cpol_grid = pyart.io.read_grid(pydda.tests.EXAMPLE_RADAR1)

sounding = pyart.io.read_arm_sonde(
    pydda.tests.SOUNDING_PATH)

# Load sounding data and insert as an intialization
u_init, v_init, w_init = pydda.initialization.make_wind_field_from_profile(
    cpol_grid, sounding[1], vel_field='corrected_velocity')

temp = u_init.shape
wind_shape = (3, temp[0], temp[1], temp[2])

# Start the wind retrieval. This example only uses the mass continuity
# and data weighting constraints.
ti = time.time()

# NOTE THAT THE FOLLOWING MUST MATCH ORDERS
prec_vectors = [[1], [2], [1,2]]
alg_list = ['single', 'double', 'dynamic']
#######

tolerances = [1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8]
subprob_tol = 1e-6

memory_list = [5, 10, 20, 30]

first_folder = '/projects/clancy/trophy_data/'
#first_folder = '/Users/clancy/repos/trophy/python/data/'
first_folder += 'runs_using_subprob_tol' + "{:.0e}".format(subprob_tol) + '/'
if not os.path.isdir(first_folder):
    cmd = 'mkdir ' + first_folder
    os.system(cmd)


# we loop through different memory levels (do lowest first)
for memory_size in memory_list:
    u_init, v_init, w_init = pydda.initialization.make_wind_field_from_profile(
        cpol_grid, sounding[1], vel_field='corrected_velocity')
    second_folder = first_folder + 'memory' + str(memory_size) + '/'
    if not os.path.isdir(second_folder):
        os.system('mkdir ' + second_folder)

    n_cumulative = [0,0,0]
    # next, step through different tolerances, once again, let most accurate, tightest bounds
    # go last. Maybe end at 1e-8 initially then add additional precisions if necessary
    for ii, tol in enumerate(tolerances):
        third_folder = second_folder + 'gtol' + "{:.0e}".format(tol) + '/'
        if not os.path.isdir(third_folder):
            os.system('mkdir ' + third_folder)

        # Finally, step through the different
        for jj, (prec_vec, alg) in enumerate(zip(prec_vectors, alg_list)):
            # if we've already solved a looser tolerance, use it for the warm start
            if ii > 0:
                warm_start_folder = second_folder + 'gtol' + "{:.0e}".format(tolerances[ii-1]) + '/' + alg + '/'
                if len(prec_vec) == 1:
                    #temp = pd.read_csv(warm_start_folder + 'summary.csv')
                    #n_prev_evals = temp['fevals']
                    warm_start_file = warm_start_folder + 'winds_and_gradient.csv'
                else:
                    #not sure if this is best, but use single start for dynamic
                    warm_start_file = warm_start_folder + 'winds_and_gradient_at_switch.csv'

                temp = pd.read_csv(warm_start_file)
                temp = np.asarray(temp['winds'])
                temp = np.reshape(temp, wind_shape)
                u_init = temp[0]
                v_init = temp[1]
                w_init = temp[2]

            fourth_folder = third_folder + alg + '/'
            if not os.path.isdir(fourth_folder):
                os.system('mkdir ' + fourth_folder)
            Grids = pydda.retrieval.get_dd_wind_field([berr_grid, cpol_grid], u_init, v_init, w_init, Co=1.0, Cm=1500.0,
                                Cz=0, frz=5000.0, filt_iterations=0, mask_outside_opt=True, upper_bc=1,
                                store_history=True, max_iterations=5000,
                                max_memory=memory_size, use_dynTR=True, gtol=tol, precision_vector=prec_vec,
                                subproblem_tol=subprob_tol, write_folder=fourth_folder)

tf = time.time()
print('Time elapsed is', tf-ti)

