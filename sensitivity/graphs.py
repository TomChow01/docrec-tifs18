'''
USAGE python sensitivity.graphs

Seaborn version 0.9.0
'''
import numpy as np
import matplotlib.pyplot as plt
import json
from matplotlib.ticker import FuncFormatter, FixedLocator
from matplotlib import rc
rc('text', usetex=True)
rc('font',**{'family':'serif','serif':['Times']})
import pandas as pd
import seaborn as sns
sns.set(
    context='paper', style='darkgrid', palette='deep', font_scale=1.0
)

results = json.load(open('sensitivity/results.json'))
df = pd.DataFrame(
    columns=('dataset', 'document', 'param', 'value', 'run', 'accuracy')
)
index = 0
dataset = []
document = []
param = []
value = []
run = []
accuracy = []

for result in results.values():
    dataset += 10 * [result['dataset']]
    document += 10 * [result['document']]
    param += 10 * [result['param']]
    value += 10 * [result['value']]
    run += list(range(1, 11))
    accuracy += result['accuracies']

df = pd.DataFrame(
    {
        'dataset': dataset,
        'document': document,
        'param': param,
        'value': value,
        'run': run,
        'accuracy': accuracy
    }
)
data = df.groupby(['param', 'value', 'run']).mean().groupby(['param', 'value']).agg(['mean', 'std'])

fig, axs = plt.subplots(ncols=3, sharey=True, figsize=(12,3), dpi=100)
handles = []
for ax, param, val, label in zip(
    axs,
    ['min_height_factor', 'max_height_factor', 'max_separation_factor'],
    [1.8, 5.5, 1.2],
    ['${H_t}_{min} (R)$', '${H_t}_{max} (R)$', '${D_c}_{max} (R)$']
):
    mean = data.loc[param][('accuracy', 'mean')]
    std = data.loc[param][('accuracy', 'std')]
    values = list(mean.index)

    def format_fn(tick_val, tick_pos):
        try:
            return '{:.2f}'.format(tick_val)
        except ValueError:
            return
    ax.xaxis.set_major_locator(FixedLocator(locs=values))
    ax.xaxis.set_major_formatter(FuncFormatter(format_fn))

    g = ax.errorbar(values, mean, yerr=std, fmt='-o', capsize=2)
    handles.append(g)
    ax.vlines(val, 0.75, 0.85, linestyle='dashed', linewidth=1)
    ax.set_xlabel(label)
    ax.set_ylim([0.75,0.85])

axs[0].set_ylabel('accuracy')

fig.tight_layout()
plt.savefig('sensitivity/graphs.pdf', bbox_inches='tight')
plt.show()
