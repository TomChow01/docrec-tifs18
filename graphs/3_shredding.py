'''
USAGE python -m graphs.3_shredding.py
'''
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
rc('text', usetex=True)
rc('font',**{'family':'serif','serif':['Times']})

import pandas as pd
import pickle
import seaborn as sns
from docrec.validation.config.experiments import ExperimentsConfig

sns.set(context='paper', style='darkgrid', font_scale=1.5)
colors = sns.color_palette('deep')
order = [0, 2, 9, 1, 4, 5, 6, 7, 8, 3]
pallete = [colors[i] for i in order]
sns.set_palette(pallete)

# dataset 1 (category info)
categories = pickle.load(open('categories_D1-mec.pkl', 'rb'))
doc_category_map = {}
for category, docs in categories.items():
    for doc in docs:
        doc_category_map[doc] = category.upper()

# global configuration
config = ExperimentsConfig('experiments.cfg', 'dataset.cfg', 'algorithms.cfg')

# columns of interest for the graphs
columns = ['dataset', 'shredding', 'method', 'document', 'accuracy']

# proposed method
df_filename = 'test/test_proposed.csv'
df = pd.read_csv(df_filename, sep='\t', encoding='utf-8')
df = df[~ np.isnan(df.accuracy)]
#df = df[(df.shredding == 'mec') & (df.solver == 'proposed')]
df = df[df.solver == 'proposed']
df['method'] = 'proposed'

keys = ['dataset', 'shredding', 'document', 'run']
df_prop = df.loc[df.groupby(keys)['nwords'].idxmax(), columns] # Note: reset_index() not need (index in uniquely defined)

# others
df_filename = 'test/test_others.csv'
df = pd.read_csv(df_filename, sep='\t', encoding='utf-8')
df = df[~ np.isnan(df.accuracy)]
df.method.replace({'andalo2': 'andalo1'}, inplace=True)

df_opt = df[df.solver == 'proposed'] # optimal solver

# andalo original
df_filename = 'test/test_andalo.csv'
df_andalo = pd.read_csv(df_filename, sep='\t', encoding='utf-8')
df_andalo = df_andalo[~ np.isnan(df_andalo.accuracy)]
#df_andalo = df_andalo[df_andalo.shredding == 'mec']
df_andalo['method'] = 'orig-andalo'

# marques original
df_filename = 'test/test_marques.csv'
df_marques = pd.read_csv(df_filename, sep='\t', encoding='utf-8')
df_marques = df_marques[~ np.isnan(df_marques.accuracy)]
#df_marques = df_marques[df_marques.shredding == 'mec']
df_marques['method'] = 'orig-marques'

df_opt = pd.concat([df_opt, df_andalo, df_marques], ignore_index=True, sort=True)

keys = ['method', 'dataset', 'shredding', 'document']

df_others = df_opt.loc[df_opt.groupby(keys)['accuracy'].idxmax(), columns]
#df_best_d1 = df_best[df_best.dataset == 'D1']
#df_best_d2 = df_best[df_best.dataset == 'D2']
#df_best['dataset'] = 'D1 + D2'
#data2 = pd.concat([df_best, df_best_d1, df_best_d2])

data = pd.concat([df_prop, df_others], ignore_index=True)

methods = ['proposed', 'marques', 'andalo1', 'morandell', 'balme', 'sleit', 'orig-andalo', 'orig-marques']
#legend = ['\\textbf{Proposed}', 'Concorde/Marques', 'Concorde/Andal\\\'o', 'Concorde/Morandell', 'Concorde/Balme', 'Concorde/Sleit', 'Andal\\\'o', 'Marques']
legend = ['\\textbf{Proposed}', 'Concorde/Marques', 'Concorde/Andal\\\'o', 'Concorde/Morandell', 'Concorde/Balme', 'Concorde/Sleit', 'Andal\\\'o', 'Marques']
legend_map = dict(zip(methods, legend))#{'proposed', 'marques', 'andalo1', 'morandell', 'balme', 'sleit'}
data.method.replace(legend_map, inplace=True)
data['shredding'] = data.shredding.map({'mec': 'mechanical', 'art': 'artificial'})

fig, axs = plt.subplots(ncols=1, figsize=(8, 4), dpi=100)
fp = sns.catplot(
    x='method', y='accuracy', order=legend, data=data,
    hue='shredding', kind='box',# size=2.5,
    margin_titles=True, fliersize=2,# width=0.5,# linewidth=0.5,
    legend=True, ax=axs
)

axs.legend_.remove()
#axs.set_aspect(aspect=4)

plt.close(fp.fig)
#axs[0].set_ylabel('')
plt.setp(axs.get_xticklabels(), rotation=20)
fig.tight_layout()
legend = plt.legend(title='shredding', bbox_to_anchor=(0.11, 0.24), loc=2, borderaxespad=0., fontsize=12)
plt.setp(axs.legend_.get_title(),fontsize=12)
plt.savefig('graphs/g3.pdf', bbox_inches='tight')
#plt.show()

