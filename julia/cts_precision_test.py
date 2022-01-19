import csv
import os
import numpy as np
git_repos = os.getenv('GITREPOS')

import sys
sys.path.append(git_repos + '/trophy/python')
import trophy
from numpy.linalg import norm
from julia import Main
import time
import pycutest_for_trophy as pycutest
import pandas as pd
np.set_printoptions(threshold=np.inf, precision=10, suppress=True,linewidth=100)


# call Prani's file that Richie amended to create problem instances with variable
Main.include(git_repos + "/trophy/julia/juliaCUTEstModule.jl")

#'''
# define a wrapper function that makes everything nice and talks to Julia module
def obj_and_grad(problem, precision, x):
    """
    :params problem: which problem to use as an objective
    :params precision: set number of bits to use for objective and gradient
    :params x: argument for objective
    :retun f, g: objective value and gradient vector at point x
    """
    Main.x = x
    Main.problem = problem
    Main.precision = precision
    obj, grad = Main.eval("juliaCUTEstModule.obj_and_grad(problem, precision, x)")
    return np.float64(obj), np.asarray(grad, dtype='float64')
#'''

# define dictionary and problem list for work later (these are problems of dimension less than 500)
prob_dict = {'HIMMELBB':2, 'CLIFF':2, 'ERRINRSM':50, 'BEALE':2, 'VARDIM':200, 'STRTCHDV':10, 'HILBERTB':10, 'JENSMP':2, 'ROSZMAN1LS':4, 'HIMMELBH':2, 'CERI651DLS':7, 'DJTL':2, 'ENGVAL2':3, 'MUONSINELS':1, 'HATFLDFLS':3, 'ZANGWIL2':2, 'GROWTHLS':3, 'SINEVAL':2, 'TRIGON2':10, 'GAUSS2LS':8, 'LSC1LS':3, 'ERRINROS':50, 'RECIPELS':3, 'QING':100, 'LUKSAN14LS':98, 'PRICE4':2, 'PALMER6C':8, 'LANCZOS2LS':6, 'BENNETT5LS':3, 'BROWNAL':200, 'GULF':3, 'PALMER2C':8, 'HUMPS':2, 'YFITU':3, 'BARD':3, 'HAIRY':2, 'MISRA1BLS':2, 'HATFLDGLS':25, 'CERI651ELS':7, 'DENSCHNC':2, 'CHWIRUT2LS':3, 'CHNRSNBM':50, 'DANWOODLS':2, 'HATFLDD':3, 'NELSONLS':3, 'JUDGE':2, 'PENALTY2':200, 'HIMMELBCLS':2, 'POWERSUM':4, 'ROSENBRTU':2, 'DENSCHND':3, 'POWELLSQLS':2, 'MARATOSB':2, 'ARGTRIGLS':200, 'LSC2LS':3, 'MGH10LS':3, 'WATSON':12, 'PALMER1C':8, 'BROWNBS':2, 'SENSORS':100, 'SSI':3, 'MISRA1DLS':2, 'TRIGON1':10, 'EXP2':2, 'S308':2, 'CERI651CLS':7, 'POWELLBSLS':2, 'MISRA1ALS':2, 'GAUSSIAN':3, 'LUKSAN17LS':100, 'ALLINITU':4, 'WAYSEA1':2, 'HATFLDE':3, 'LUKSAN11LS':100, 'RAT43LS':4, 'SNAIL':2, 'DANIWOODLS':2, 'VESUVIOLS':8, 'RAT42LS':3, 'CHNROSNB':50, 'KIRBY2LS':5, 'BRKMCC':2, 'LUKSAN21LS':100, 'HIMMELBG':2, 'LUKSAN16LS':100, 'SPIN2LS':102, 'PALMER5C':6, 'ARGLINC':200, 'MANCINO':100, 'THURBERLS':7, 'CLUSTERLS':2, 'PALMER4C':8, 'LANCZOS1LS':6, 'DEVGLA2':5, 'CHWIRUT1LS':3, 'HELIX':3, 'PALMER3C':8, 'CERI651ALS':7, 'LANCZOS3LS':6, 'WAYSEA2':2, 'EXPFIT':2, 'DEVGLA1':4, 'BOX3':3, 'VIBRBEAM':30, 'MEXHAT':2, 'ECKERLE4LS':3, 'PALMER8C':8, 'BROWNDEN':4, 'LOGHAIRY':2, 'HILBERTA':2, 'CERI651BLS':7, 'ELATVIDU':2, 'DENSCHNB':2, 'MGH17LS':5, 'DENSCHNF':2, 'DENSCHNA':2, 'PALMER1D':7, 'GAUSS3LS':8, 'OSBORNEA':5, 'HIMMELBF':4, 'PRICE3':2, 'VESUVIOULS':8, 'MGH09LS':4, 'BIGGS6':6, 'CUBE':2, 'LUKSAN15LS':100, 'KOWOSB':4, 'HATFLDFL':3, 'PALMER7C':8, 'PALMER5D':4, 'VESUVIALS':8, 'GAUSS1LS':8, 'LUKSAN12LS':98, 'ROSENBR':2, 'BOXBODLS':2, 'EGGCRATE':2, 'MISRA1CLS':2, 'DENSCHNE':3, 'STREG':4, 'PENALTY3':200}
prob_dict.pop('DANWOODLS')      # DANWOODLS is getting a negative number for a logarithm
prob_dict.pop('VIBRBEAM')       # VIBRBEAM seems to have a different dimension between Prani and pycutest implementation
prob_dict.pop('ECKERLE4LS')     # division by zero
prob_dict.pop('MISRA1CLS')      # returns NaNs
prob_dict.pop('OSBORNEA')       # returns NaNs
prob_dict.pop('PALMER1D')       # got a segmentation fault running code    
prob_dict.pop('NELSONLS')       # got seg fault 
prob_dict.pop('MGH10LS')       # got seg fault 

# the following are problems where discrepancies were observed (over 3 seconds, rel single err>1e-6, rel double err > 1e-8, single double grad angle > 1e-6, or gradient angle greater than 1e-6)
prob_dict.pop('STRTCHDV')
prob_dict.pop('ROSZMAN1LS')
prob_dict.pop('CERI651DLS')
prob_dict.pop('DJTL')
prob_dict.pop('TRIGON2')
prob_dict.pop('CERI651ELS')
prob_dict.pop('ROSENBRTU')
prob_dict.pop('ARGTRIGLS')
prob_dict.pop('CERI651CLS')
prob_dict.pop('VESUVIOLS')
prob_dict.pop('SPIN2LS')
prob_dict.pop('HELIX')
prob_dict.pop('CERI651ALS')
prob_dict.pop('CERI651BLS')
prob_dict.pop('VESUVIOULS')
prob_dict.pop('VESUVIALS')
prob_dict.pop('PENALTY3')
prob_list = list(prob_dict.keys())

# for now, set two different precision dictionaries to see how the algorithms compare

# construct precision dictionary
prec_dict = {11:1}                                                     # half
#prec_dict = {24:1}                                                     # single
#prec_dict = {53:1}                                                     # double
#prec_dict = {24:1, 53:2}                                               # singe, double
#prec_dict = {11:1, 24:2, 53:3}                                         # half, single, double
#prec_dict = {8:1, 11:2, 17:3, 24:4, 53:5}                             # other precision types
#prec_dict = {8:1, 13:2, 18:3, 23:4, 28:5, 33:6, 38:7, 43:8, 48:9, 53:10}      # every 5 bits 
#prec_dict = {10:1, 14:2, 15:3}

# construct names to use for the different levels of precision
tmp = ''
for k in prec_dict.keys():
    tmp = tmp + str(k) + '_'
tmp = tmp[:-1]

readwrite_file = 'data/julia_prec_' + tmp + '.csv'
if not os.path.isfile(readwrite_file):
    f = open(readwrite_file, 'a')
    f.close()


print(os.stat(readwrite_file).st_size)
if os.stat(readwrite_file).st_size == 0:
    preclist = [str(i) for i in prec_dict.keys()]
    columns = ['problem','prob_dim','funcvals','gradnorm','success','nits']
    columns.extend(preclist)
    print(columns)
    with open(readwrite_file, mode='w') as tmp:
        tmp_writer = csv.writer(tmp, delimiter=',')
        tmp_writer.writerow(columns)
    

temp = pd.read_csv(readwrite_file)
if len(temp) > 0:
    last_problem = temp.iloc[-1,0]
else:
    last_problem = 'problem'


found = False
for i, name in enumerate(prob_list):
    print(name, last_problem)
    if name == last_problem:
        print('problem found, starting at problem number', i)
        found = True
        break
if not found:
    print('Problem not found, starting from the beginning')
    i = -1

prob_list = prob_list[(i+1):]
small_dict = {}
for p in prob_list:
    small_dict[p] = prob_dict[p]    
prob_dict = small_dict


# initialize variables
eps = 1.0e-5
epsTR = 1.0e-6
max_memory = 15
max_problem_dim = 100

funvals1 = list()
gradnorms1 = list()
success1 = list()
nits1 = list()
preccounter1 = list(list() for ii in range(len(prec_dict)))


def run_all():
    maxit = 1000
    # loop through all problems in problem dictionary
    for i, (prob, prob_dim) in enumerate(prob_dict.items()):
        print("\n\n", str(i) + ". Working on problem", prob, "of dim", prob_dim)
        if prob_dim <= max_problem_dim:
            print(prob)
            prob_list.append(prob)
            #"""
            # define function handles to pass to algorithm
            
            func = lambda z, nbits: obj_and_grad(prob, nbits, z) 
            p_double = pycutest.import_problem(prob+"_double")

            # get starting point
            x0 = np.asarray(p_double.x0)
            
            ti = time.time()
            # call TROPHY using Prani's functions with continuously variable precision
            ret1 = trophy.DynTR(x0, func, prec_dict, gtol=eps, max_iter=maxit, 
                    tr_tol=epsTR, verbose=True, max_memory=max_memory, norm_to_use=2, gamma_dec=0.25)
            tpr = time.time() - ti
            funvals1.append(ret1.fun)
            gradnorms1.append(norm(ret1.jac))
            success1.append(ret1.success)
            nits1.append(ret1.nit)
            for i, v in enumerate(list(ret1.precision_counts.values())):
                preccounter1[i].append(v)

            #time.sleep(5)
            #df1 = pd.DataFrame(data=funvals1, columns=['funcvals'])
            df1 = pd.DataFrame(data=[prob], columns=['problem'])
            df1['prob_dim'] = prob_dim
            df1['funcvals'] = ret1.fun
            df1['gradnorm'] = norm(ret1.jac)
            df1['success'] = ret1.success
            df1['nits'] = ret1.nit
            for (k,v) in ret1.precision_counts.items():
                df1[str(k)] = v
            
            
            print(df1)

            if i == 1:
                use_header=True
            else:
                use_header = False

            df1.to_csv(readwrite_file, mode='a', header=use_header, index=False)

        else:
            print('Problem too big')



if __name__=="__main__":
    run_all()

