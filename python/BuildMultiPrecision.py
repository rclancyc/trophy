
import numpy as np
import sys
import os

# pull home directory and location of py
home_dir = os.environ['HOME']
pycutest_cache = os.environ['PYCUTEST_CACHE']
func_loc = pycutest_cache + '/pycutest_cache_holder/'

# add the location of pycutest_cache_holder to the path
sys.path.append(pycutest_cache)
sys.path.append('./pycutest_for_trophy/')

# import augmented pycutest package for single and double precision builds
import pycutest_for_trophy as pycutest

# find all unconstrained problems for which SIF files exist
probs = pycutest.find_problems(constraints='U')

# get current directory then change directory to location where different precision functions will be placed.
curr_dir = os.getcwd()
os.chdir(func_loc)

bad_problems = ['JIMACK']
problems_not_built = list()

# loop through all available unconstrained problems
for (i, problem) in enumerate(probs):

    # problems with hyphen in their name seem to fail...not sure why
    if '-' not in problem and problem not in bad_problems:
        print(i + 1, '. Importing problem', problem)
        # initially, all problems will be empty, so run once, preduce, relabel folder, then run again
        if not (os.path.exists(func_loc + problem+"_single")):
            p = pycutest.import_problem(problem, precision="single")
            cmd = 'mv ' + problem + ' ' + problem + '_single'
            os.system(cmd)
        else:
            print(problem+"_single already cached")
        if not (os.path.exists(problem+"_double")):
            p = pycutest.import_problem(problem, precision="double")
            cmd = 'mv ' + problem + ' ' + problem + '_double'
            os.system(cmd)
        else:
            print(problem+"_double already cached")
    else:
        print(problem, "is broken, don't import")
        problems_not_built.append(problem)

os.chdir(curr_dir)