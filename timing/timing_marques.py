'''
USAGE python -m timing.timing_parallel
'''
import sys
import os
import numpy as np
import pickle
from time import time
import pandas as pd
from docrec.strips.strips import Strips
from docrec.reconstruction.solver.sdhp import \
    shortest_directed_hamiltonian_path as solve
from docrec.reconstruction.compatibility.marques import Marques
from docrec.validation.config.experiments import ExperimentsConfig
from docrec.ocr.recognition import number_of_words
from multiprocessing import Pool, cpu_count


t0_glob = time()

# global configuration
config = ExperimentsConfig('experiments.cfg', 'dataset.cfg', 'algorithms.cfg')

# algorithm parameters
params = config.algorithms_config.algorithms['marques'].params

# sample document
doc = 'dataset/D1/mechanical/D014'
margins = pickle.load(open('margins_D1-mec.pkl', 'rb'))
left = margins['D014']['left']
right = margins['D014']['right']


def compute_solution(run):
    tload = time()
    strips = Strips(path=doc, filter_blanks=False)
    strips.trim(left, right)
    tload = time() - tload

    tcomp = time()
    alg = Marques(strips)
    mat = alg(d=3, **params).matrix
    tcomp = time() - tcomp

    topt = time()
    sol, cost = solve(mat)
    topt = time() - topt

    result = {
        'run': run,
        'load': tload,
        'comp': tcomp,
        'opt': topt,
        'total': tload + tcomp + topt
    }
    return result

# Resulting dataframe

df = pd.DataFrame(columns=('run', 'load', 'comp', 'opt', 'total'))
nrows = 10
index = 1
for run in range(1, 11): # 10 runs
    result = compute_solution(run)
    elapsed = time() - t0_glob
    done = 100 * float(index) / nrows
    estimated = ((elapsed / index) * (nrows - index))
    run = result['run']
    tload = result['load']
    tcomp = result['comp']
    topt = result['opt']
    total = result['total']
    print('[{:.2f}% - {}/{}] (estimated {:.3f}s)'.format(done, index, nrows, estimated))
    df.loc[index] = [run, tload, tcomp, topt, total]
    index += 1

# dumping
df_filename = 'timing/timing_marques.csv'
df.to_csv(df_filename, sep='\t', encoding='utf-8', index=False)
