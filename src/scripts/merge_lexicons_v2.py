"""
Lexicons must be in csv format, with first column 'WORD' second column as numerical polarity
"""

import pandas as pd
import click
from src.utils import Logger, __fn__, list_files
from functools import reduce

logger = Logger(__fn__())


lx_dir = 'data/processed/lexicon_v2'
outfile = 'lexicon_table_v2'

lx_files = list_files(lx_dir)
extensions = set([f.split('.')[-1] for f in lx_files])
assert len(extensions) ==1, 'Invalid files format {}, only accept .csv'.format(extensions)

read_files = [f for f in lx_files if f.strip('.csv') != outfile]
path = lx_dir + '/' if lx_dir[-1] not in {'/','\\'} else lx_dir

# mpqa = pd.read_csv(path+read_files[0], usecols=range(2), dtype={'WORD': str}, index_col='WORD')
# senti = pd.read_csv(path+read_files[3], usecols=range(2), dtype={'WORD': str}, index_col='WORD')


logger.info('Reading lexicons: {}'.format(read_files))
dframes = [pd.read_csv(path + f, usecols=range(2), dtype={'WORD': str}) for f in read_files]
logger.info('Merging lexicons with shapes: {}'.format([df.shape for df in dframes]))
df = reduce(lambda x,y: pd.merge(x, y, how='outer', on='WORD'), dframes)
logger.info('Writting merged lexicon to {}, merged shape: {}'.format(path, df.shape))
logger.info('Duplidates: {}'.format(df.duplicated().value_counts().loc[True]))
df = df.drop_duplicates()
logger.info('Shape after dropping duplicates: {}'.format(df.shape))
df = df.groupby('WORD').mean()
logger.info('Shape after groupby: {}'.format(df.shape))
logger.info('Filling NaN with mean polarity')
df = df.apply(lambda s: s.fillna(s.mean()), axis=1)

# Add
to_remove = ['try', 'bar', 'too']
to_add = {'not': -1, "n't": -1}

for w in to_remove:
    df.drop(w, inplace=True)

_df = df.transpose()
for k,v in to_add.items():
    _df[k] = v
df = _df.transpose()

df.to_csv(path + outfile + '.csv', index=True)
logger.info('Merged lexicon saved to {}'.format(path + outfile + '.csv'))

