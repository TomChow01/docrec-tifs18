'''
USAGE python test_marques.py
'''
import sys
import os
import numpy as np
from time import clock
import pickle
import gc
import pandas as pd
#from utils import check_matrix
from docrec.strips.strips import Strips
from docrec.validation.metrics.solution import accuracy
from docrec.reconstruction.compatibility.marques import Marques
from docrec.validation.config.experiments import ExperimentsConfig
from docrec.reconstruction.solver.nn import nearest_neighbor

# global configuration
config = ExperimentsConfig('experiments.cfg', 'dataset.cfg', 'algorithms.cfg')

# solver
solve = lambda matrix: nearest_neighbor(matrix)

# datasets
datasets = config.dataset_config.datasets
del datasets['training']
ndocs_total = sum([len(dataset.docs) for dataset in datasets.values()])

# algorithm params
params = config.algorithms_config.algorithms['marques'].params

# resulting dataframe
index = index_backup = 0
df = pd.DataFrame(
    columns=('method', 'dataset', 'shredding', 'document', 'disp', 'accuracy', 'cost', 'solution', 'time')
)
df_filename = 'test/test_marques.csv'
if os.path.exists(df_filename):
    index_backup = len(pd.read_csv(df_filename, sep='\t'))
else:
    df.to_csv(df_filename, sep='\t', encoding='utf-8', index=False)

ndisp = config.ndisp
total = ndocs_total * ndisp

for dataset in datasets.values():
    # skip dataset
    ndocs = len(dataset.docs)
    if index + ndocs * ndisp <= index_backup:
        index += ndocs * ndisp
        continue

    dset_main_id, dset_shredding = dataset.id.split('-')
    for doc in dataset.docs:
        # skip document
        if index + ndisp <= index_backup:
            index += ndisp
            continue

        print('Loadind strips of {} ... '.format(doc.id), end='')
        sys.stdout.flush()
        strips = Strips(path=doc.path, filter_blanks=False)
        t0 = clock()
        if dataset.id != 'D2-mec':
            margins = pickle.load(open('margins_{}.pkl'.format(dataset.id), 'rb'))
            left = margins[doc.id]['left']
            right = margins[doc.id]['right']
            strips.trim(left, right)
        t_strips = clock() - t0
        print('Elapsed time: {:.2f}s'.format(t_strips))
        sys.stdout.flush()

        print('Building algorithm object for {} ... '.format(doc.id), end='')
        sys.stdout.flush()
        t0 = clock()
        alg = Marques(strips)
        t_alg = clock() - t0
        print('Elapsed time: {:.2f}s'.format(t_alg))
        sys.stdout.flush()

        for disp in range(ndisp):
            done = 100 * float(index + 1) / total
            print('    [{:.2f}%] dataset={} doc={} disp={}'.format(done, dataset.id, doc.id, disp))
            sys.stdout.flush()

            print('    Matrix ... ', end='')
            sys.stdout.flush()
            t0 = clock()
            matrix = alg(d=disp, **params).matrix
            t_mat = clock() - t0
            print('    Elapsed time: {:.2f}s'.format(t_mat))
            sys.stdout.flush()

            print('    Solution ... ', end='')
            sys.stdout.flush()
            t0 = clock()
            sol, cost = solve(matrix)
            acc = accuracy(sol)
            sol = ' '.join(str(v) for v in sol)
            t_sol = clock() - t0
            print('    Elapsed time: {:.2f}s'.format(t_sol))

            # storing
            t = t_strips + t_alg + t_mat + t_sol
            df.loc[index] = ['marques', dset_main_id, dset_shredding, doc.id, disp, acc, cost, sol, t]
            index += 1

        # dumping
        df.to_csv(df_filename, sep='\t', encoding='utf-8', index=False, mode='a', header=False)
        df.drop(df.index, inplace=True) # clear

        # memory management
        gc.collect()
        print('{} items in garbage'.format(len(gc.garbage)))
        sys.stdout.flush()
        # sys.exit()