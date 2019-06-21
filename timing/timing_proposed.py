'''
USAGE python -m timing.timing_parallel
'''
import sys
import os
import numpy as np
from time import time
import pandas as pd
from docrec.strips.stripschar import StripsChar
from docrec.reconstruction.solver.sdhp import \
    shortest_directed_hamiltonian_path as solve
from docrec.reconstruction.compatibility.proposed import Proposed
from docrec.validation.config.experiments import ExperimentsConfig
from docrec.ocr.recognition import number_of_words
from multiprocessing import Pool, cpu_count


t0_glob = time()

# global configuration
config = ExperimentsConfig('experiments.cfg', 'dataset.cfg', 'algorithms.cfg')

# algorithm parameters
params = config.algorithms_config.algorithms['proposed'].params

# sample document
doc = 'dataset/D1/mechanical/D014'

def compute_solution(run):
    tseg = time()
    strips = StripsChar(path=doc, filter_blanks=True)
    tseg = time() - tseg

    tcomp = time()
    seed = config.seed + run
    alg = Proposed(strips, seed=seed, verbose=True, trailing=3)
    matrix = alg(**params).matrix
    tcomp = time() - tcomp

    topt = time()
    solutions = []
    #costs = []
    for mat in matrix:
        sol, cost = solve(mat)
        solutions.append(sol)
        #costs.append(cost)
    topt = time() - topt

    tocr = time()
    for sol in solutions:
        number_of_words(strips.image(sol), 'pt_BR')
    tocr = time() - tocr

    result = {
        'seg': tseg,
        'comp': tcomp,
        'opt': topt,
        'ocr': tocr,
        'total': tseg + tcomp + topt + tocr
    }
    return result

# Resulting dataframe

df = pd.DataFrame(columns=('run', 'seg', 'comp', 'opt', 'ocr', 'total'))
nrows = 10
index = 1
for run in range(1, 11): # 10 runs
    result = compute_solution(run)
    elapsed = time() - t0_glob
    done = 100 * float(index) / nrows
    estimated = ((elapsed / index) * (nrows - index))
    tseg = result['seg']
    tcomp = result['comp']
    topt = result['opt']
    tocr = result['ocr']
    total = result['total']
    print('[{:.2f}% - {}/{}] (estimated {:.3f}s)'.format(done, index, nrows, estimated))
    df.loc[index] = [run, tseg, tcomp, topt, tocr, total]
    index += 1

# dumping
df_filename = 'timing/timing_proposed.csv'
df.to_csv(df_filename, sep='\t', encoding='utf-8', index=False)
