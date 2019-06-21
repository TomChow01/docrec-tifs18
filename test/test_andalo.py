'''
USAGE python test_andalo.py
'''
import sys
import os
import numpy as np
from time import clock
import pickle
import gc
import pandas as pd
from utils import check_matrix
from docrec.strips.strips import Strips
from docrec.validation.metrics.solution import accuracy
from docrec.reconstruction.jigsaw.jigsaw import solve_from_matrix
from docrec.reconstruction.compatibility.andalo import Andalo
from docrec.validation.config.experiments import ExperimentsConfig

# global configuration
config = ExperimentsConfig('experiments.cfg', 'dataset.cfg', 'algorithms.cfg')

# solver
solve = lambda matrix: solve_from_matrix(matrix)

# datasets
datasets = config.dataset_config.datasets
del datasets['training']
ndocs_total = sum([len(dataset.docs) for dataset in datasets.values()])

# algorithms
params1 = config.algorithms_config.algorithms['andalo1'].params
params2 = config.algorithms_config.algorithms['andalo2'].params

# resulting dataframe
index = index_backup = 0
df = pd.DataFrame(
    columns=('method', 'dataset', 'shredding', 'document', 'disp', 'accuracy', 'cost', 'solution', 'time')
)
df_filename = 'test/test_andalo.csv'
if os.path.exists(df_filename):
    index_backup = len(pd.read_csv(df_filename, sep='\t'))
else:
    df.to_csv(df_filename, sep='\t', encoding='utf-8', index=False)

ndisp = config.ndisp
total = 2 * ndocs_total * ndisp

for dataset in datasets.values():
    # skip dataset
    ndocs = len(dataset.docs)
    if index + 2 * ndocs * ndisp <= index_backup:
        index += 2 * ndocs * ndisp
        continue

    dset_main_id, dset_shredding = dataset.id.split('-')
    for doc in dataset.docs:
        # skip document
        if index + 2 * ndisp <= index_backup:
            index += 2 * ndisp
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
        alg = Andalo(strips)
        t_alg = clock() - t0
        print('Elapsed time: {:.2f}s'.format(t_alg))
        sys.stdout.flush()

        for disp in range(ndisp):
            done = 100 * float(index + 1) / total
            print('    [{:.2f}%] dataset={} doc={} disp={}'.format(done, dataset.id, doc.id, disp))
            sys.stdout.flush()

            print('    Matrix 1 ... ', end='')
            sys.stdout.flush()
            t0 = clock()
            matrix1 = alg(d=disp, **params1).matrix
            t_mat1 = clock() - t0
            print('    Elapsed time: {:.2f}s'.format(t_mat1))
            sys.stdout.flush()

            print('    Matrix 2 ... ', end='')
            sys.stdout.flush()
            t0 = clock()
            matrix2 = alg(d=disp, **params2).matrix
            t_mat2 = clock() - t0
            print('    Elapsed time: {:.2f}s'.format(t_mat2))
            sys.stdout.flush()

            print('    Solution 1 ... ', end='')
            sys.stdout.flush()
            t0 = clock()
            acc1 = float('nan')
            sol1 = None
            cost1 = float('nan')
            if check_matrix(matrix1):
                sol1, cost1 = solve(matrix1)
                if sol1 is not None:
                    acc1 = accuracy(sol1)
            sol1 = ' '.join(str(v) for v in sol1) if sol1 is not None else ''
            t_sol1 = clock() - t0
            print('    Elapsed time: {:.2f}s'.format(t_sol1))

            print('    Solution 2 ... ', end='')
            sys.stdout.flush()
            t0 = clock()
            acc2 = float('nan')
            sol2 = None
            cost2 = float('nan')
            if check_matrix(matrix2):
                sol2, cost2 = solve(matrix2)
                if sol2 is not None:
                    acc2 = accuracy(sol2)
            sol2 = ' '.join(str(v) for v in sol2) if sol2 is not None else ''
            t_sol2 = clock() - t0
            print('    Elapsed time: {:.2f}s'.format(t_sol2))

            # storing
            t1 = t_strips + t_alg + t_mat1 + t_sol1
            t2 = t_strips + t_alg + t_mat2 + t_sol2
            df.loc[index] = ['andalo1', dset_main_id, dset_shredding, doc.id, disp, acc1, cost1, sol1, t1]
            df.loc[index + 1] = ['andalo2', dset_main_id, dset_shredding, doc.id, disp, acc2, cost2, sol2, t2]
            index += 2

        # dumping
        df.to_csv(df_filename, sep='\t', encoding='utf-8', index=False, mode='a', header=False)
        df.drop(df.index, inplace=True) # clear

        # memory management
        gc.collect()
        print('{} items in garbage'.format(len(gc.garbage)))
        sys.stdout.flush()
        # sys.exit()