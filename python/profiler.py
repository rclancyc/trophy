import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import copy



def profiler(data_set):
    """
    for the profiler, we need the name of the problem (maybe although probably not) the 
    field we are concerned with, i.e., num. its., and whether or not it was a success for 
    EACH solver. So maybe the best way is to construct it beforehand. 

    suppose our data_set had problems by row and solver by column.
    :param data_set:
    """
    s = data_set
    num_problems, num_solvers = s.shape

    # assumes that data set has infinite values if solver did not solve it
    s_min = np.min(s, axis=1)
    if min(s_min) < 0: 
        print('negative numbers present, this is a problem')
    if min(s_min) == 0:
        print('profiling something with a zero values')
        bad_idx = s_min == 0
        s[bad_idx,:] = s[bad_idx,:] + 1
        s_min[bad_idx] = 1    
    # check to see if there are problems where none of the solvers solved it
    unsolved_problems_idx = (s_min == np.inf)

    # this will set unsolved problem rows to -1 which indicates this it was unsolved problem
    # we only do this so it doesn't throw and error
    s[unsolved_problems_idx,:] = -1

    # get ratios
    r = (s.T/s_min).T

    # set unsolved problems to infinity (since dividing -1 by np.inf will give 0 or -0)
    r[unsolved_problems_idx,:] = np.inf
    
    # tells use what x values to use for our profiler
    xs = np.unique(r)

    ys = np.zeros((xs.shape[0], num_solvers))
    # construct cumulative sum for each different solver
    for j in range(num_solvers):
            # step through unique values of r (points used for x values in profiler)
            for i, val in enumerate(xs):
                idx = r[:,j] < val
                ys[i,j] = sum(idx)/num_problems
    
    # return x values to print along x-axis for profiler and y values where each column is a different solver
    return xs, ys



def slice_and_dice(data_list, field):
    """
    This will take a list of data from different solvers and a paricular field, then return a matrix with cleaned data
    """

    num_solvers = len(data_list)
    num_problems = len(data_list[0])

    # we want an array with entries of value of interest and each column is for a different solver
    arr  = np.zeros((num_problems, num_solvers))
    for ii, db in enumerate(data_list): 
        temp = copy.copy(db.loc[:, field])
        fail_idx = np.logical_or(db.loc[:,('success')] == "failed", db.loc[:,('success')] == False)   # store values of whehter
        temp.loc[fail_idx] = np.inf                     # for failed problems (ones not solved), set to infinity
        arr[:, ii] = temp                               # store column for current solver


    # return cleaned data with infinite values for failed iterations
    return arr
    






eps = 1.0e-4
max_problem_dim = 26
file_path = '/Users/clancy/repos/trophy/python/data/eps0.0001max_vars26/'
sin_path = file_path + 'single_max'+str(max_problem_dim)+ '_eps'+ str(eps) + 'vars.csv'
dou_path = file_path + 'double_max'+str(max_problem_dim)+ '_eps'+ str(eps) + 'vars.csv'
dyn_path = file_path + 'dynTR_max'+str(max_problem_dim)+ '_eps'+ str(eps) + 'vars.csv'

legends = ['Single TR', 'Double TR', 'TROPHY']

sin = pd.read_csv(sin_path)
dou = pd.read_csv(dou_path)
dyn = pd.read_csv(dyn_path)

solver_list = ['Single TR', 'Double TR', 'TROPHY'] #, 'L-BFGS', 'TR-NCG']

db_list = [sin, dou, dyn]
for df in db_list:
    # treats single evals as half the cost of double
    df['adjusted_evals'] = (df['fevals'] - df['single_evals']) + 0.5*df['single_evals']

db_list = [sin, dou, dyn]
fields = ['time','feval', 'gradnorm', 'nits', 'fevals', 'sing_evals', 'adjusted_evals']

use_field = ['adjusted_evals', 'nits', 'time', 'gradnorm']

ii = 0
for field in use_field:
    ii += 1
    arr = slice_and_dice(db_list, field)
    xs, ys = profiler(arr)
    plt.subplot(2,2,ii)
    plt.semilogx(xs, ys, label=legends)
    plt.xlim([1, xs[-3]])
    plt.ylim([0,1])
    plt.title(use_field[ii-1])
    plt.grid(True)

plt.legend(loc='lower right')
plt.show()





