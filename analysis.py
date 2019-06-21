'''
USAGE python analysis.py
'''
import math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle

# dataset 1 (category info)
categories = pickle.load(open('categories_D1-mec.pkl', 'rb'))
doc_category_map = {}
for category, docs in categories.items():
    for doc in docs:
        doc_category_map[doc] = category.upper()
print('#documents in D1 (per categories): TO={} LG={} FG={}'.format(
    len(categories['to']), len(categories['lg']), len(categories['fg'])
))

# Data to be analyzed
df_filename = 'test/test_proposed.csv'
df = pd.read_csv(df_filename, sep='\t', encoding='utf-8')
df = df[~ np.isnan(df.accuracy)]
df['shredding'] = df.shredding.map({'mec': 'mechanical', 'art': 'artificial'})

df_prop = df[df.solver == 'proposed']
keys = ['dataset', 'shredding', 'document', 'run']

df_prop_glob = df_prop.copy()
df_best_glob = df_prop_glob.loc[df_prop_glob.groupby(keys)['nwords'].idxmax()].reset_index()

# accuracy for mechanical shredding
df_ = df_best_glob[df_best_glob['shredding'] == 'mechanical']
print('accuracy for mechanical shredding (proposed)')
print('=> overall={:.3f}'.format(df_['accuracy'].mean()))
print('=> per dataset')
print(df_.groupby('dataset')['accuracy'].mean().round(3))
print('=> per categories in D1')
df_ = df_[df_['dataset'] == 'D1']
df_['category'] = df_['document'].map(doc_category_map)
print(df_.groupby(['category'])['accuracy'].mean().round(3))

# decay




# # print('Accuracy for D1 (TO)={:.3f}'.format(df_mean_d1_mec_to.accuracy.mean()))
# # print('Accuracy decay for D1={:.3f}'.format(df_mean_d1_art.accuracy.mean() - df_mean_d1_mec.accuracy.mean()))
# # print('Accuracy decay for D1 (TO)={:.3f}'.format(df_mean_d1_art_to.accuracy.mean() - df_mean_d1_mec_to.accuracy.mean()))
# # print('Accuracy decay for D1 (LG)={:.3f}'.format(df_mean_d1_art_lg.accuracy.mean() - df_mean_d1_mec_lg.accuracy.mean()))
# # print('Accuracy decay for D1 (FG)={:.3f}'.format(df_mean_d1_art_fg.accuracy.mean() - df_mean_d1_mec_fg.accuracy.mean()))

# df_best_d1 = df_best_glob[df_best_glob.dataset == 'D1'].copy()
# df_best_d2 = df_best_glob[df_best_glob.dataset == 'D2'].copy()
# # #df_best_glob['dataset'] = 'D1 + D2'

# # #data = pd.concat([df_best_glob, df_best_d1, df_best_d2])

# keys = ['document']
# df_mean_d1_mec = df_best_d1[df_best_d1.shredding == 'mechanical'].groupby(keys)['accuracy'].mean().reset_index(name='accuracy')
# df_mean_d1_art = df_best_d1[df_best_d1.shredding == 'artificial'].groupby(keys)['accuracy'].mean().reset_index(name='accuracy')
# df_mean_d2_mec = df_best_d2[df_best_d2.shredding == 'mechanical'].groupby(keys)['accuracy'].mean().reset_index(name='accuracy')
# df_mean_d2_art = df_best_d2[df_best_d2.shredding == 'artificial'].groupby(keys)['accuracy'].mean().reset_index(name='accuracy')

# df_mean_d1_mec['category'] = df_mean_d1_mec.document.map(doc_category_map)
# df_mean_d1_art['category'] = df_mean_d1_art.document.map(doc_category_map)

# df_mean_d1_mec_to = df_mean_d1_mec[df_mean_d1_mec.category == 'TO']
# df_mean_d1_art_to = df_mean_d1_art[df_mean_d1_art.category == 'TO']
# df_mean_d1_mec_lg = df_mean_d1_mec[df_mean_d1_mec.category == 'LG']
# df_mean_d1_art_lg = df_mean_d1_art[df_mean_d1_art.category == 'LG']
# df_mean_d1_mec_fg = df_mean_d1_mec[df_mean_d1_mec.category == 'FG']
# df_mean_d1_art_fg = df_mean_d1_art[df_mean_d1_art.category == 'FG']

# print('Accuracy for D1 (TO)={:.3f}'.format(df_mean_d1_mec_to.accuracy.mean()))
# print('Accuracy decay for D1={:.3f}'.format(df_mean_d1_art.accuracy.mean() - df_mean_d1_mec.accuracy.mean()))
# print('Accuracy decay for D1 (TO)={:.3f}'.format(df_mean_d1_art_to.accuracy.mean() - df_mean_d1_mec_to.accuracy.mean()))
# print('Accuracy decay for D1 (LG)={:.3f}'.format(df_mean_d1_art_lg.accuracy.mean() - df_mean_d1_mec_lg.accuracy.mean()))
# print('Accuracy decay for D1 (FG)={:.3f}'.format(df_mean_d1_art_fg.accuracy.mean() - df_mean_d1_mec_fg.accuracy.mean()))

# df_mean_d1_mec_to_la = df_mean_d1_mec_to.copy().sort_values(['accuracy'])
# df_mean_d1_mec_lg_la = df_mean_d1_mec_lg.copy().sort_values(['accuracy'])
# df_mean_d1_mec_fg_la = df_mean_d1_mec_fg.copy().sort_values(['accuracy'])

# # uncoment the following lines to show the the accuracy by document
# #print('Sorted accuracy for mechanicall shredding')
# #print(df_mean_d1_mec_to_la)
# #print(df_mean_d1_mec_lg_la)
# #print(df_mean_d1_mec_fg_la)

# print('Accuracy for D2={:.3f}'.format(df_mean_d2_mec.accuracy.mean()))
# print('Accuracy decay for D2={:.3f}'.format(df_mean_d2_art.accuracy.mean() - df_mean_d2_mec.accuracy.mean()))

# keys = ['dataset', 'shredding', 'document', 'run']
# df_best_acc_glob = df_prop_glob.loc[df_prop_glob.groupby(keys)['accuracy'].idxmax()].reset_index()
# p = float((df_best_acc_glob.accuracy == df_best_glob.accuracy).sum()) / len(df_best_acc_glob)
# print('OCR-aided filter effectiveness={:.3f}'.format(100 * p))

# # not used
# #print('#words for well-chosen solutions')
# #print(df_best_glob[df_best_acc_glob.accuracy == df_best_glob.accuracy].nwords.mean())

# #print('#words for badly-chosen solutions')
# #print(df_best_glob[df_best_acc_glob.accuracy != df_best_glob.accuracy].nwords.mean())

# df_diff = (df_mean_d1_art.accuracy - df_mean_d1_mec.accuracy).reset_index(name='accuracy')
# df_diff['document'] = df_mean_d1_art.document
# df_diff['category'] = df_diff.document.map(doc_category_map)

# df_bad = df_diff[df_diff.accuracy > 0.2]
# print('#documents for which accuracy decay more than 20%={}/60 ({:.3f}%)'.format(len(df_bad), 100 * len(df_bad) / 60.0))

# uncoment the following lines to show the the accuracy by document
# df_bad_to = df_bad[df_bad.category == 'TO']
# df_bad_lg = df_bad[df_bad.category == 'LG']
# df_bad_fg = df_bad[df_bad.category == 'FG']
# n_to = len(categories['to'])
# n_lg = len(categories['lg'])
# n_fg = len(categories['fg'])
# print('TO={}/{}'.format(len(df_bad_to), n_to))
# print('LG={}/{}'.format(len(df_bad_lg), n_lg))
# print('FG={}/{}'.format(len(df_bad_fg), n_fg))

df_filename = 'test/test_others.csv'
df = pd.read_csv(df_filename, sep='\t', encoding='utf-8')
df = df[~ np.isnan(df.accuracy)]
df.method.replace({'andalo2': 'andalo1'}, inplace=True)
df = df[df['shredding'] == 'mec']# = df.shredding.map({'mec': 'mechanical', 'art': 'artificial'})
df = df[df.solver == 'proposed']

df = df.loc[df.groupby(['method', 'dataset', 'shredding', 'document'])['accuracy'].idxmax()]#.reset_index()

print('Accuracy for mechanical shredding (others)')
print(df.groupby('method')['accuracy'].mean().round(3))

print('Accuracy for mechanical shredding D1 - TO (others)')
#df_diff['category'] = df_diff.document.map(doc_category_map)
#df _= df[df]
df_ = df[df['dataset'] == 'D1'].copy()
df_['category'] = df_['document'].map(doc_category_map)
print(df_.groupby(['method', 'category'])['accuracy'].mean().round(3))

print('Accuracy for mechanical shredding D2 (others)')
df_ = df[df['dataset'] == 'D2'].copy()
print(df_.groupby(['method'])['accuracy'].mean().round(3))

# df_ = df_others_best_glob[df_others_best_glob.shredding == 'mechanical']
# df_ = df_[df_['dataset'] == 'D1']
# print(df_)
#    [((df_others_best_glob.shredding == 'mechanical') & ()) | (() & ())]
#print(df_others_best_glob[df_others_best_glob.shredding == 'mechanical'].groupby(['method', 'dataset'])['accuracy'].mean().round(3))


#print('Accuracy for mechanical shredding={:.3f}'.format(df_others_best_glob[df_others_best_glob.shredding == 'mechanical'].accuracy.mean()))
#print(df_others_best_glob.document)
#print(len(df))
# print df_others_glob
# #df_best_glob['dataset'] = 'D1 + D2'
# #df_others_best_glob['dataset'] = 'D1 + D2'
# df_best_glob['method'] = 'proposed'
# #df_others_best_glob.method.replace({'andalo2': 'andalo1'}, inplace=True)

# data = pd.concat([df_best_glob, df_others_best_glob])
# keys = ['method', 'shredding']
# print 'Global results'
# print data.groupby(keys)['accuracy'].mean()

# keys = ['method', 'dataset', 'shredding']
# print 'Results by dataset'
# print data.groupby(keys)['accuracy'].mean()

# keys = ['method', 'shredding', 'category']
# data = data[data.dataset == 'D1']
# data['category'] = data.document.map(doc_category_map)
# print 'Results by category (D1)'
# print data.groupby(keys)['accuracy'].mean()

# # (2) Time analysis
# df_filename = config.path_cache('timing.csv')
# df = pd.read_csv(df_filename, sep='\t', encoding='utf-8')
# df = df[~ np.isnan(df.accuracy)]
# df['shredding'] = df.shredding.map({'mec': 'mechanical', 'art': 'artificial'})




