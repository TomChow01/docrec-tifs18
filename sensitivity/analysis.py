'''
USAGE python sensitivity.analysis

Seaborn version 0.9.0
'''
import json
import pandas as pd

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

# range of +/- 20%
print('range of +/- 20%')
data = df.groupby(['param', 'value', 'run']).agg(['mean'])
print('average accuracy={:.3f}'.format(float(data.mean())))
print('std accuracy={:.3f}'.format(float(data.std())))
print('minimum accuracy={:.3f}'.format(float(data.min())))
print('maximum accuracy={:.3f}'.format(float(data.max())))
print('variation={:.2f}%'.format(100 * float(data.max() - data.min())))

# range of +/- 10% (40% less samples)
min_height_factor = 1.8
min_height_factor_min = 0.9 * min_height_factor
min_height_factor_max = 1.1 * min_height_factor

max_height_factor = 5.5
max_height_factor_min = 0.9 * max_height_factor
max_height_factor_max = 1.1 * max_height_factor

max_separation_factor = 1.2
max_separation_factor_min = 0.9 * max_separation_factor
max_separation_factor_max = 1.1 * max_separation_factor

df = df[~((df['param'] == 'min_height_factor') & (df['value'] < min_height_factor_min))]
df = df[~((df['param'] == 'min_height_factor') & (df['value'] > min_height_factor_max))]
df = df[~((df['param'] == 'max_height_factor') & (df['value'] < max_height_factor_min))]
df = df[~((df['param'] == 'max_height_factor') & (df['value'] > max_height_factor_max))]
df = df[~((df['param'] == 'max_separation_factor') & (df['value'] < max_separation_factor_min))]
df = df[~((df['param'] == 'max_separation_factor') & (df['value'] > max_separation_factor_max))]

data = df.groupby(['param', 'value', 'run']).agg(['mean'])
print('range of +/- 10%')
print('average accuracy={:.3f}'.format(float(data.mean())))
print('std accuracy={:.3f}'.format(float(data.std())))
print('minimum accuracy={:.3f}'.format(float(data.min())))
print('maximum accuracy={:.3f}'.format(float(data.max())))
print('variation={:.2f}%'.format(100 * float(data.max() - data.min())))