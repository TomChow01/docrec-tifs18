'''
USAGE python -m timing.analysis
'''
import sys
import os
import pandas as pd
from docrec.strips.stripschar import StripsChar

# Sample document
doc = 'dataset/D1/mechanical/D014'
#strips = StripsChar(path=doc, filter_blanks=True)
#print('# of characters= {}'.format(len(strips.inner)))

# proposed
df_filename = 'timing/timing_proposed.csv'
df = pd.read_csv(df_filename, sep='\t', encoding='utf-8')

M = int(df.total.mean() / 60)
S = int(df.total.mean() % 60)
print('Average time (proposed)={:.2f}s ({}m{}s)'.format(df.total.mean(), M, S))
agg = df.agg('mean')
print('seg={:.2f}% comp={:.2f}% opt={:.2f}% ocr={:.2f}%'.format(
    100 * agg['seg'] / agg['total'], 100 * agg['comp'] / agg['total'], 100 * agg['opt'] / agg['total'], 100 * agg['ocr'] / agg['total']
))

df_filename = 'timing/timing_parallel.csv'
df = pd.read_csv(df_filename, sep='\t', encoding='utf-8')
M = int(df.total.mean() / 60)
S = int(df.total.mean() % 60)
print('Average time (proposed parallel)={:.2f}s ({}m{}s)'.format(df.total.mean(), M, S))

df_filename = 'timing/timing_marques.csv'
df = pd.read_csv(df_filename, sep='\t', encoding='utf-8')
print('Average time (marques)={:.2f}s)'.format(df.total.mean()))