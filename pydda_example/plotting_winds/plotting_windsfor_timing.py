import numpy as np
import pyart
import sys
import os
import time
import pandas as pd
from datetime import datetime

# change to location of trophy repo
repo_path = os.environ['HOME'] + '/repos/'
sys.path.insert(0, repo_path + 'trophy/python/')
sys.path.insert(0, repo_path + 'trophy/python/dynTRpydda_for_timing/')


import dynTRpydda_for_timing as pydda

from matplotlib import pyplot as plt
import numpy
import warnings
warnings.filterwarnings("ignore")


#numpy.show_config()

berr_grid = pyart.io.read_grid(pydda.tests.EXAMPLE_RADAR0)
cpol_grid = pyart.io.read_grid(pydda.tests.EXAMPLE_RADAR1)

sounding = pyart.io.read_arm_sonde(
    pydda.tests.SOUNDING_PATH)


# Load sounding data and insert as an intialization
u_init, v_init, w_init = pydda.initialization.make_wind_field_from_profile(
    cpol_grid, sounding[1], vel_field='corrected_velocity')
aa, bb, cc = u_init.shape

#u_init = u_init + np.random.normal(0, 1, u_init.shape)
#v_init = v_init + np.random.normal(0, 1, v_init.shape)
#w_init = w_init + np.random.normal(0, 1, w_init.shape)
#winds = pd.read_csv("/Users/clancy/repos/trophy/python/warm_start.csv")
#winds = pd.read_csv("/Users/clancy/repos/trophy/python/warm_start2.csv")
#winds = np.array(winds.iloc[:,1])
#winds = np.asarray(winds.iloc[:,0])

# should be 1 through 12


tol = 1e0
subprob_tol = 1e-8
memory_size = 10
wind_idx = 9

try:
    winds = pd.read_csv("/Users/clancy/repos/trophy/python/wind_matrix.csv")
except:
    try:
        winds = pd.read_csv("/home/clancy/repos/trophy/python/wind_matrix.csv")
    except:
        winds = pd.read_csv("/home/rclancy/repos/trophy/python/wind_matrix.csv")
winds = np.asarray(winds.iloc[:, wind_idx])
winds = np.reshape(winds, (3, aa, bb, cc))
u_init = winds[0]
v_init = winds[1]
w_init = winds[2]



precisions = {'half': 1, 'single': 2, 'double': 3}
#precisions = {'single': 1, 'double': 2}
#precisions = {'half': 0}
#precisions = {'single': 1}
#precisions = {'double': 2}


dt_string = 'run' + str(wind_idx)
alg = ''
for key in precisions.keys():
    alg += '_' + key
alg = alg[1::]
if os.path.isdir('/home/clancy'):
    third_folder = '/home/clancy/trophy_data/'+alg+'/'
if os.path.isdir('/home/rclancy'):
    third_folder = '/home/rclancy/trophy_data/'+alg+'/'
if os.path.isdir('/Users/clancy'):
    third_folder = '/Users/clancy/trophy_data/'+alg+'/'

if not os.path.isdir(third_folder):
    os.system('mkdir ' + third_folder)

detail_str = 'memory'+str(memory_size) + '_subprobtol' + "{:.0e}".format(subprob_tol) + '/'
if len(precisions) > 1:
    detail_str = 'tol' + "{:.0e}".format(tol) + '_' + detail_str

fourth_folder = third_folder + detail_str
if not os.path.isdir(fourth_folder):
    os.system('mkdir ' + fourth_folder)

fourth_folder += dt_string + '/'
if not os.path.isdir(fourth_folder):
    os.system('mkdir ' + fourth_folder)


ti = time.time()
Grids = pydda.retrieval.get_dd_wind_field([berr_grid, cpol_grid], u_init, v_init, w_init, Co=1.0, Cm=1500.0,
                                          Cz=0, frz=5000.0, filt_iterations=0, mask_outside_opt=True, upper_bc=1,
                                          store_history=True, max_iterations=50000,
                                          max_memory=memory_size, use_dynTR=True, gtol=tol, precision_dict=precisions,
                                          subproblem_tol=subprob_tol, write_folder=fourth_folder)
tf = time.time()
print('Time elapsed is', tf-ti)