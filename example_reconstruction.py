import numpy as np
import sys
import cv2
import pandas as pd
from docrec.strips.strips import Strips
from docrec.validation.config.experiments import ExperimentsConfig

config = ExperimentsConfig('experiments.cfg', 'dataset.cfg', 'algorithms.cfg')

df_filename = 'test/test_proposed.csv'
df = pd.read_csv(df_filename, sep='\t', encoding='utf-8')
df = df[~ np.isnan(df.accuracy)]
df_prop = df[(df.solver == 'proposed') & (df.dataset == 'D1') & (df.shredding == 'mec')]#.reset_index()

keys = ['run']
for doc in ['D002', 'D009', 'D057']:
    df_doc = df_prop[df_prop.document == doc]
    mean = df_doc.loc[df_doc.groupby(keys)['nwords'].idxmax()].accuracy.mean() # mean of 10 runs
    acc = df_doc.loc[(df_doc.accuracy - mean).abs().idxmin()].accuracy # closest accuracy wrt the average acc
    print('doc={} acc={:.3f}'.format(doc, acc))
    sol = df_doc.loc[(df_doc.accuracy - mean).abs().idxmin()].solution # solution with closest acc wrt to the avg acc
    sol = [int(v) for v in sol.split()]
    strips = Strips(path='dataset/D1/mechanical/{}'.format(doc))
    img = strips.image(order=sol)
    cv2.imwrite(
        'results/{}_rec.jpg'.format(doc),
        cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    )
