import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import copy
git_repos = os.getenv('GITREPOS')



def profiler(data_set):
    """
    for the profiler, we need the name of the problem (maybe although probably not) the 
    field we are concerned with, i.e., num. its., and whether or not it was a success for 
    EACH solver. So maybe the best way is to construct it beforehand. 

    suppose our data_set had problems by row and solver by column.
    :param data_set:
    """
    s = data_set
    num_solvers = s.shape[1]

    s_min = np.min(s, axis=1)

    # assumes that data set has infinite values if solver did not solve it
    delete_unsolved = False
    if delete_unsolved:
        
        get_rid_of_idx = s_min < np.inf
        s = s[get_rid_of_idx,:]
        s_min = s_min[get_rid_of_idx]

    if min(s_min) < 0: 
        print('negative numbers present, this is a problem')
    if min(s_min) == 0:
        print('profiling something with a zero values')
        bad_idx = s_min == 0
        s[bad_idx,:] = s[bad_idx,:] + 1
        s_min[bad_idx] = 1    
    # check to see if there are problems where none of the solvers solved it
    
    if sum(s_min == np.inf) > 0:
        unsolved_problems_idx = (s_min == np.inf)

        # this will set unsolved problem rows to -1 which indicates this it was unsolved problem
        # we only do this so it doesn't throw and error
        s[unsolved_problems_idx,:] = -1

    # get ratios
    r = (s.T/s_min).T

    if sum(s_min == np.inf) > 0:
    # set unsolved problems to infinity (since dividing -1 by np.inf will give 0 or -0)
        r[unsolved_problems_idx,:] = np.inf
    
    # tells use what x values to use for our profiler
    xs = np.unique(r)

    ys = np.zeros((xs.shape[0], num_solvers))
    num_problems = r.shape[0]

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
    





#eps = 0.0001
eps = 1e-05
#eps = 1e-06
max_problem_dim = 101
max_num_bits = 53

file_path = git_repos + 'trophy/julia/data/'

prec_level = ['11', '24', '53', '24_53', '11_24_53', '8_11_17_24_53', '8_13_18_23_28_33_38_43_48_53']
legends = [   'H',  'S',  'D',  'S,D',   'H,S,D',    '8,11,17,24,53', 'Every 5 bits']

file_list = list()
db_list = list()
for pl in prec_level:
    file_name = file_path + 'julia_prec_' + pl + '.csv'
    file_list.append(file_name)

    df = pd.read_csv(file_name)
    colnames = df.columns
    col_idx = [c.isdigit() for c in colnames]
    sub_df = df.iloc[:, col_idx]
    sub_df.columns

    # get the relative cost based on number of bits
    n_bits = np.array([int(i) for i in sub_df.columns])
    rel_cost = n_bits/max_num_bits

    data_mat = np.asarray(sub_df)
    bits_complexity = data_mat@rel_cost
    bits_complexity[np.logical_not(df['success'])] = np.inf
    df['adjusted_evals'] = bits_complexity
    db_list.append(df)




solver_list = legends




fields = ['time','feval', 'gradnorm', 'nits', 'fevals', 'sing_evals', 'adjusted_evals']

use_field = ['adjusted_evals', 'nits', 'time', 'gradnorm']
use_title = ['Adjusted values', 'Number of iterations','Time', 'Gradient norm']

plt.figure(figsize=(8,6))

hfont = {'fontname':'Times', 'fontsize':20}
afont = {'fontname':'Times', 'fontsize':14}

ii = 0
print(eps)
for field in use_field:
    ii += 1
    arr = slice_and_dice(db_list, field)
    xs, ys = profiler(arr)
    plt.subplot(2,2,ii)
    #plt.subplot(2,2,ii)
    plt.semilogx(xs, ys[:,0], label='Single TR', linestyle='-.')
    plt.semilogx(xs, ys[:,1], label='Double TR', linestyle='--')
    plt.semilogx(xs, ys[:,2], label='TROPHY (proposed)', linestyle='-')
    plt.xscale('log', base=2)
    plt.xlim([1, xs[-4]])
    plt.ylim([0,1])
    plt.xticks(**afont)
    plt.yticks(**afont)
    if ii % 2 == 0:
        plt.yticks(color='w', **hfont)
    plt.title(use_title[ii-1], **hfont)
    plt.grid(True)

plt.suptitle('Tolerance ' + str(eps))
plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=.25, hspace=.35)
plt.legend(loc='lower right')
plt.show()





