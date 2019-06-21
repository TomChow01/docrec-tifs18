'''
USAGE python -m graphs.1_proposed.py
'''
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from matplotlib import rc
rc('text', usetex=True)
rc('font',**{'family':'serif','serif':['Times']})
import pandas as pd
import pickle
import seaborn as sns

from docrec.validation.config.experiments import ExperimentsConfig


sns.set(
    context='paper', style='darkgrid', palette='deep', font_scale=1.8
)
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

# data to be analyzed
df_filename = 'test/test_proposed.csv'
df = pd.read_csv(df_filename, sep='\t', encoding='utf-8')
df = df[~ np.isnan(df.accuracy)]
df['shredding'] = df.shredding.map({'mec': 'mechanical', 'art': 'artificial'})

df_solver = df[df.solver == 'proposed']
keys = ['dataset', 'shredding', 'document', 'run']

df_prop_glob = df_solver.copy()
df_prop_best_glob = df_prop_glob.loc[df_prop_glob.groupby(keys)['nwords'].idxmax()].reset_index()

df_prop_best_d1 = df_prop_best_glob[df_prop_best_glob.dataset == 'D1'].copy()
df_prop_best_d2 = df_prop_best_glob[df_prop_best_glob.dataset == 'D2'].copy()
df_prop_best_glob['dataset'] = 'D1 + D2'

data_prop = pd.concat([df_prop_best_glob, df_prop_best_d1, df_prop_best_d2])
data_prop.drop(['s'], axis=1, inplace=True)
data_prop['method'] = 'proposed'

fig, axs = plt.subplots(ncols=2, sharey=True, figsize=(8, 4), dpi=100)

fp1 = sns.catplot(
    x='dataset', y='accuracy', data=data_prop,
    hue='shredding', kind='box', #height=1.15,# aspect=1.5,
    margin_titles=True,# fliersize=2,# width=0.65, linewidth=1.0,
    legend=False, ax=axs[0]
)

axs[0].legend_.remove()
#axs[0].set_aspect(aspect=4)

df_prop_best_d1['category'] = df_prop_best_d1.document.map(doc_category_map)
fp2 = sns.catplot(
    x='category', y='accuracy', data=df_prop_best_d1,
    hue='shredding', kind='box', #height=1.15,# aspect=1.5,
    order=['TO', 'LG', 'FG'], margin_titles=True,# fliersize=2,
    #width=0.65, linewidth=1.0,
    legend=False, legend_out=False, ax=axs[1]
)
axs[1].legend_.remove()
#axs[1].set_aspect(aspect=1.5)
plt.close(fp1.fig)
plt.close(fp2.fig)
axs[1].set_ylabel('')

plt.close(fp1.fig)
plt.close(fp2.fig)
axs[1].set_ylabel('')

fig.tight_layout()
legend = plt.legend(title='shredding', bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize=12)
plt.setp(legend.get_title(),fontsize=12)
plt.savefig('graphs/g1.pdf', bbox_inches='tight')
# plt.show()