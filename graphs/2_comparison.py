'''
USAGE python graphs/2_comparison.py
'''
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
rc('text', usetex=True)
rc('font',**{'family':'serif','serif':['Times']})

import pandas as pd
import pickle
import seaborn as sns
sns.set(context='paper', style='darkgrid', font_scale=1.7)
sns.set_palette(sns.color_palette('Set2', n_colors=8))
#sns.set_palette(sns.color_palette('muted'))

from docrec.validation.config.experiments import ExperimentsConfig


# dataset 1 (category info)
categories = pickle.load(open('categories_D1-mec.pkl', 'rb'))
doc_category_map = {}
for category, docs in categories.items():
    for doc in docs:
        doc_category_map[doc] = category.upper()

# global configuration
config = ExperimentsConfig('experiments.cfg', 'dataset.cfg', 'algorithms.cfg')

# columns of interest for the graphs
columns = ['dataset', 'method', 'document', 'accuracy']

# proposed method
df_filename = 'test/test_proposed.csv'
df = pd.read_csv(df_filename, sep='\t', encoding='utf-8')
df = df[~ np.isnan(df.accuracy)]
df = df[(df.shredding == 'mec') & (df.solver == 'proposed')]
df['method'] = 'proposed'

keys = ['dataset', 'document', 'run']
df_best = df.loc[df.groupby(keys)['nwords'].idxmax(), columns] # Note: reset_index() not need (index in uniquely defined)
df_best_d1 = df_best[df_best.dataset == 'D1']
df_best_d2 = df_best[df_best.dataset == 'D2']
df_best['dataset'] = 'D1 + D2'
data1 = pd.concat([df_best, df_best_d1, df_best_d2])

# others
df_filename = 'test/test_others.csv'
df = pd.read_csv(df_filename, sep='\t', encoding='utf-8')
df = df[~ np.isnan(df.accuracy)]
df.method.replace({'andalo2': 'andalo1'}, inplace=True)
df = df[df.shredding == 'mec']

df_opt = df[df.solver == 'proposed'] # optimal solver

# andalo original
df_filename = 'test/test_andalo.csv'
df_andalo = pd.read_csv(df_filename, sep='\t', encoding='utf-8')
df_andalo = df_andalo[~ np.isnan(df_andalo.accuracy)]
df_andalo = df_andalo[df_andalo.shredding == 'mec']
df_andalo['method'] = 'orig-andalo'

# marques original
df_filename = 'test/test_marques.csv'
df_marques = pd.read_csv(df_filename, sep='\t', encoding='utf-8')
df_marques = df_marques[~ np.isnan(df_marques.accuracy)]
df_marques = df_marques[df_marques.shredding == 'mec']
df_marques['method'] = 'orig-marques'

df_opt = pd.concat([df_opt, df_andalo, df_marques], ignore_index=True, sort=True)

keys = ['method', 'dataset', 'document']

df_best = df_opt.loc[df_opt.groupby(keys)['accuracy'].idxmax(), columns]
df_best_d1 = df_best[df_best.dataset == 'D1']
df_best_d2 = df_best[df_best.dataset == 'D2']
df_best['dataset'] = 'D1 + D2'
data2 = pd.concat([df_best, df_best_d1, df_best_d2])

data = pd.concat([data1, data2], ignore_index=True)

methods = ['proposed', 'marques', 'andalo1', 'morandell', 'balme', 'sleit', 'orig-andalo', 'orig-marques']
legend = ['\\textbf{Proposed}', 'Concorde/Marques', 'Concorde/Andal\\\'o', 'Concorde/Morandell', 'Concorde/Balme', 'Concorde/Sleit', 'Andal\\\'o', 'Marques']
legend_map = dict(zip(methods, legend))#{'proposed', 'marques', 'andalo1', 'morandell', 'balme', 'sleit'}
data.method.replace(legend_map, inplace=True)
#print(legend_map)
#print(zip(methods, legend))
fig, axs = plt.subplots(ncols=2, sharey=True, figsize=(16, 4), dpi=100)

fp1 = sns.catplot(
    x='dataset', y='accuracy', data=data,
    hue='method', hue_order=legend, kind='box', #size=1.5,
    margin_titles=True, fliersize=2,# width=0.65, linewidth=0.5,
    legend=False, ax=axs[0],
)
axs[0].legend_.remove()
#axs[0].set_aspect(aspect=1.5)

data_d1 = data[data.dataset == 'D1'].copy()
data_d1['category'] = data_d1.document.map(doc_category_map)

fp2 = sns.catplot(
    x='category', y='accuracy', order=['TO', 'LG', 'FG'], hue_order=legend,
    data=data_d1,
    hue='method', kind='box',# size=1.5,
    margin_titles=True, fliersize=2, #width=0.65, linewidth=0.5,
    legend=False, ax=axs[1]
)

axs[1].legend_.remove()
#axs[1].set_aspect(aspect=1.5)
plt.close(fp1.fig)
plt.close(fp2.fig)
axs[1].set_ylabel('')

fig.tight_layout()
legend = plt.legend(title='method', bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize=12)
plt.setp(legend.get_title(),fontsize=12)
plt.savefig('graphs/g2.pdf', bbox_inches='tight')
# plt.show()