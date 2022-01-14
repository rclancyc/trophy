import numpy as np
import pandas as pd

import pycutest_for_trophy as pycutest
import trophy
import sys
import os
import time
from util_func import pycutest_wrapper
from numpy.linalg import norm
from scipy.optimize import SR1

# add pycutestcache path if you haven't done so elsewhere
temp = os.environ['PYCUTEST_CACHE']
cache_dir = temp + '/pycutest_cache_holder/'
sys.path.append(temp)
sys.path.append(cache_dir)

def test_all():

    # initialize variables
    eps = 1.0e-5
    epsTR = 1.0e-6
    maxit = 5000
    max_problem_dim = 101
    max_memory = 15
    singleTR = list()
    doubleTR = list()
    #lbfgs = list()
    #trncg = list()
    dynTR = list()

    # create new directory if there isn't one
    use_dir = 'data/eps'+str(eps)+'max_vars'+str(max_problem_dim)
    if not (os.path.exists(use_dir)):
        os.mkdir(use_dir)


    # specify largest problem to solve then 
    sing_file = use_dir + '/single_max'+str(max_problem_dim) + '_eps'+ str(eps) + 'vars.csv'
    doub_file = use_dir + '/double_max'+str(max_problem_dim) + '_eps'+ str(eps) + 'vars.csv'
    dyn_file = use_dir + '/dynTR_max'+str(max_problem_dim) + '_eps'+ str(eps) + 'vars.csv'
    #bfgs_file = use_dir + '/lbfgs_max'+str(max_problem_dim) + '_eps'+ str(eps) + 'vars.csv'
    #trncg_file = use_dir + '/trncg_max'+str(max_problem_dim) + '_eps'+ str(eps) + 'vars.csv'

    # create data fields for write files
    fields = ['problem', 'dimension', 'success', 'time', 'feval', 'gradnorm', 'nits', 'fevals', 'single_evals','message']

    # collect unconstrained problems to work on
    problems = pycutest.find_problems(constraints='U') #, n=[1,5000])

    # loop through cutest problems  
    j = -1
    for (i, prob_str) in enumerate(problems):
        

        if prob_str not in ['DEVGLA2NE', 'JIMACK', 'S308NE'] and 'BA-' not in prob_str:

            # for problem that exist
            if os.path.exists(cache_dir+prob_str+"_single"):
                
                # construct function handles for single, double, and dynamic precision
                p1 = pycutest.import_problem(prob_str+"_single")
                p2 = pycutest.import_problem(prob_str+"_double")
                dim = p1.n

                # constuct function handles for different solvers
                func_single = lambda z, prec: p1.obj(z, gradient=True)
                func_double = lambda z, prec: p2.obj(z, gradient=True)
                func_dynamic = lambda z, prec: pycutest_wrapper(z, prec, p1, p2)
                func_bfgs = lambda z: p2.obj(z, gradient=True)
                hessian = lambda z: p2.hess(z)

                # is the problem dimension smaller than the max? 
                if p1.n <= max_problem_dim:
                    print('\n')
                    print(i+1, '. Solving problem', prob_str, " in dim=", p1.n)
                    x0 = p1.x0
                    _, curr_g = func_bfgs(x0)
                    norm_g = norm(curr_g)

                    # SINGLE PRECISION
                    print('Solving single TR.', end=' ')
                    t0 = time.time()
                    ret = trophy.DynTR(x0, func_single, {'single': 1}, gtol=eps, max_iter=maxit, tr_tol=epsTR, verbose=False, max_memory=max_memory, norm_to_use=2)
                    t_elapsed = time.time() - t0
                    mystr = '(' + ret.message[0:25] + ')'
                    print(mystr)
                    if ret.success:
                        #success = 'converged'
                        success = True
                    else:
                        #success = 'failed'
                        success = False
                    temp = [prob_str, dim, success, t_elapsed, ret.fun, norm(ret.jac), ret.nit, ret.nfev, ret.precision_counts['single'], ret.message]
                    singleTR.append(temp)


                    # DOUBLE PRECISION
                    t0 = time.time()
                    print('Solving double TR.', end=' ')
                    ret = trophy.DynTR(x0, func_double, {'double': 2}, gtol=eps, max_iter=maxit, tr_tol=epsTR, verbose=False, max_memory=max_memory, norm_to_use=2)
                    t_elapsed = time.time() - t0
                    mystr = '(' + ret.message[0:25] + ')'
                    print(mystr)
                    if ret.success:
                        #success = 'converged'
                        success = True
                    else:
                        #success = 'failed'
                        success = False
                    temp = [prob_str, dim, success, t_elapsed, ret.fun, norm(ret.jac), ret.nit, ret.nfev, 0, ret.message]
                    doubleTR.append(temp)

                    """
                    # LBFGS SOLVER
                    t0 = time.time()
                    print('Solving LBFGS.', end=' ')
                    ret = scipy.optimize.minimize(func_bfgs, x0, method="L-BFGS-B", jac=True, tol=10**(-33), bounds=None, options={'ftol': 10**(-33), 'gtol': eps, 'maxcor': max_memory, 'maxiter': maxit})
                    t_elapsed = time.time() - t0
                    mystr = '(' + ret.message[0:25] + ')'
                    print(mystr)
                    if ret.success:
                        if 'REL_RED' in ret.message:
                            success = 'failed'
                        else:
                            success = 'converged'
                    else:
                        success = 'failed'
                    temp = [prob_str, success, t_elapsed, ret.fun, norm(ret.jac), ret.nit, ret.nfev, 0, ret.message]
                    lbfgs.append(temp)

                    # SCIPY TR newton CG SOLVER
                    t0 = time.time()
                    print('Solving Trust region CG.', end=' ')
                    ret = scipy.optimize.minimize(func_bfgs, x0, method='trust-ncg', jac=True, hess=hessian, options={'gtol': eps, 'maxiter': maxit})
                    t_elapsed = time.time() - t0
                    mystr = '(' + ret.message[0:50] + ')'
                    print(mystr)
                    if ret.success:
                        if 'REL_RED' in ret.message:
                            success = 'failed'
                        else:
                            success = 'converged'
                    else:
                        success = 'failed'
                    temp = [prob_str, success, t_elapsed, ret.fun, norm(ret.jac), ret.nit, ret.nfev, 0, ret.message]
                    trncg.append(temp)
                    """

                    # DYNAMIC PRECISION
                    t0 = time.time()
                    print('Solving dynamic TR.', end=' ')
                    ret = trophy.DynTR(x0, func_dynamic, {'single': 1, 'double': 2}, gtol=eps, max_iter=maxit, tr_tol=epsTR, verbose=False, max_memory=max_memory, norm_to_use=2)
                    t_elapsed = time.time() - t0
                    mystr = '(' + ret.message[0:25] + ')'
                    print(mystr)
                    if ret.success:
                        #success='converged'
                        success = True
                    else:
                        #success='failed'
                        success = False
                    temp = [prob_str, dim, success, t_elapsed, ret.fun, norm(ret.jac), ret.nit, ret.nfev, ret.precision_counts['single'], ret.message]
                    dynTR.append(temp)

                    if np.mod(j+1, 10) == 0 or p1.n > 100:
                        sing_df = pd.DataFrame(data=singleTR, columns=fields)
                        doub_df = pd.DataFrame(data=doubleTR, columns=fields)
                        dyn_df  = pd.DataFrame(data=dynTR, columns=fields)
                        #bfgs_df = pd.DataFrame(data=lbfgs, columns=fields)
                        #trncg_df = pd.DataFrame(data=trncg, columns=fields)
                        sing_df.to_csv(sing_file)
                        doub_df.to_csv(doub_file)
                        dyn_df.to_csv(dyn_file)
                        #bfgs_df.to_csv(bfgs_file)
                        #trncg_df.to_csv(trncg_file)


                else:
                    print('\n', i+1, '. ' + prob_str + ' problem exceeds maximum number of dimensions')

            else:
                print("Can't find file " + prob_str)
    

    sing_df = pd.DataFrame(data=singleTR, columns=fields)
    doub_df = pd.DataFrame(data=doubleTR, columns=fields)
    dyn_df = pd.DataFrame(data=dynTR, columns=fields)
    #bfgs_df = pd.DataFrame(data=lbfgs, columns=fields)
    #trncg_df = pd.DataFrame(data=trncg, columns=fields)

    sing_df.to_csv(sing_file)
    doub_df.to_csv(doub_file)
    dyn_df.to_csv(dyn_file)
    #bfgs_df.to_csv(bfgs_file)
    #trncg_df.to_csv(trncg_file)



def main():
    test_all()



if __name__ == "__main__":
    main()

