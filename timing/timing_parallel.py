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
params['ns'] = 1

# sample document
doc = 'dataset/D1/mechanical/D014'

def compute_solution(s):

    mat = alg(**params).matrix[0]
    sol, cost = solve(mat)
    nwords = number_of_words(strips.image(sol), 'pt_BR')
    return None

# Resulting dataframe

df = pd.DataFrame(columns=('run', 'seg', 'pre', 'pool', 'total'))
nrows = 10
index = 1
for run in range(1, 11): # 10 runs
    tseg = time()
    strips = StripsChar(path=doc, filter_blanks=True)
    tseg = time() - tseg

    tpre = time()
    seed = config.seed + run
    alg = Proposed(strips, seed=seed, verbose=True, trailing=3)
    tpre = time() - tpre

    tpool = time()
    with Pool(processes=10) as pool:
        for result in pool.imap_unordered(compute_solution, list(range(1, 11))):
            pass
    tpool = time() - tpool

    elapsed = time() - t0_glob
    done = 100 * float(index) / nrows
    estimated = ((elapsed / index) * (nrows - index))

    total = tseg + tpre + tpool
    print('[{:.2f}% - {}/{}] (estimated {:.3f}s)'.format(done, index, nrows, estimated))
    df.loc[index] = [run, tseg, tpre, tpool, total]
    index += 1

# dumping
df_filename = 'timing/timing_parallel.csv'
df.to_csv(df_filename, sep='\t', encoding='utf-8', index=False)
