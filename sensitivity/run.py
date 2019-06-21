from multiprocessing import Pool, cpu_count
import sys
import os
import gc
import numpy as np
import random
import pandas as pd
import pickle
from time import time, sleep
from docrec.strips.stripschar import StripsChar
from docrec.validation.metrics.solution import accuracy
from docrec.reconstruction.solver.sdhp import \
    shortest_directed_hamiltonian_path as solve
from docrec.reconstruction.compatibility.proposed import Proposed
from docrec.validation.config.experiments import ExperimentsConfig
import json


t0_glob = time()

config = ExperimentsConfig('experiments.cfg', 'dataset.cfg', 'algorithms.cfg')

# processed instances
results = dict()
if os.path.exists('sensitivity/results.json'):
    results = json.load(open('sensitivity/results.json', 'r'))
    print(type(results.keys()))

# algorithm (method) parameters
params = config.algorithms_config.algorithms['proposed'].params
#params['ns'] = 1

def compute_solution(instance):
    t0 = time()
    id_ = instance['id']    
    dataset = instance['dataset'] 
    doc = instance['document']
    param = instance['param']
    value = instance['value']
    #run = instance['run']
    kwargs = {param: value}
    strips = StripsChar(path=doc.path, filter_blanks=True, **kwargs)
    seed = config.seed + int(id_)
    alg = Proposed(strips, seed=seed, verbose=True, trailing=3)
    matrix = alg(**params).matrix
    
    accuracies = []
    solutions = []
    for mat in matrix:
        sol, _ = solve(mat, id_=id_)
        acc = accuracy(sol) if sol is not None else float('nan')
        accuracies.append(acc)
        solutions.append(sol)
                
    result = {
        'id': id_,
        'dataset': dataset,        
        'document': doc.id,
        'param': param,
        'value': value,
        'accuracies': accuracies,
        'solutions': solutions,
        'time': time() - t0
    }
    return result


# investigated parameters
perc = 0.2
min_height_factor = 1.8
max_height_factor = 5.5
max_separation_factor = 1.2
min_height_factors = np.linspace(
    min_height_factor - perc * min_height_factor, min_height_factor + perc * min_height_factor, 10
)
max_height_factors = np.linspace(
    max_height_factor - perc * max_height_factor, max_height_factor + perc * max_height_factor, 10
)
max_separation_factors = np.linspace(
    max_separation_factor - perc * max_separation_factor, max_separation_factor + perc * max_separation_factor, 10
)

id_ = 1
instances = []
for dataset in ['D1-mec', 'D2-mec']:
    docs = config.dataset_config.datasets[dataset].docs
    for doc in docs:
        for param, values in zip(
            ['min_height_factor', 'max_height_factor', 'max_separation_factor'],
            [min_height_factors, max_height_factors, max_separation_factors]
        ):
            for value in values:
                if str(id_) not in results:
                    instance = {
                        'id': str(id_),
                        'dataset': dataset,    
                        'document': doc,
                        'param': param,
                        'value': value,
                    }
                    instances.append(instance)
                id_ += 1


# main loop
total = len(instances)
with Pool(processes=30) as pool:
    for i, result in enumerate(pool.imap_unordered(compute_solution, instances), 1):
        elapsed = time() - t0_glob
        done = 100 * float(i) / total
        estimated = ((elapsed / i) * (total - i)) / 3600

        # update results
        id_ = result['id']
        results[id_] = result

        # print current status
        dataset = result['dataset']
        doc = result['document']
        param = result['param']
        value = result['value']
        print('[{:.2f}% - {}/{}] dataset={} doc={} {}={} (estimated {:.3f} hs)'.format(
            done, i, total, dataset, doc, param, value, estimated
        ))
        json.dump(results, open('sensitivity/results.json', 'w'))

