from src.models import VALID_MODELS
from src.utils import Logger, __fn__, load_corpus, get_envar, read_config, get_timestamp
from src.data import AbsaDataManager, LexiconManager
import click
from sklearn.model_selection import KFold
import numpy as np
import pandas as pd
import itertools


logger = Logger(__fn__())


def train_validate(model_name, params, train_df, val_df, test_df=None):

    # Build Model
    lm = LexiconManager()
    dm = AbsaDataManager(lexicon_manager=lm)
    model = VALID_MODELS[model_name.lower()]
    model = model(datamanager=dm, parameters=params)

    # Train
    record = model.train(train_df, val_df, test_df, early_stop='acc3')

    return record

def get_acc(epoch_str, position):
    return float(epoch_str.split()[position].strip('%').split('=')[-1])/100

def best_val_acc(cv_list):
    a = [get_acc(s, 5) for s in cv_list]
    return max(a), a.index(max(a))

def best_acc(cv_list):
    """Best based on best val_acc3"""
    #val_score, e = best_val_acc(cv_list)
    e = -2
    val_score = get_acc(cv_list[e], 5)
    test_score = get_acc(cv_list[e], 7)
    train_score = get_acc(cv_list[e], 3)
    return train_score, val_score, test_score

def best_scores(cv_dict, complete=False):
    a = np.array([best_acc(l) for l in cv_dict.values()])
    df = pd.DataFrame(dict(zip(cv_dict.keys(), a)), index=['TRAIN', 'DEV', 'TEST']).transpose()
    if complete:
        return a, df
    else:
        return a.mean(axis=0), df

def product_dict(**kwargs):
    keys = kwargs.keys()
    vals = [v if isinstance(v, list) else [v] for v in kwargs.values()]
    for instance in itertools.product(*vals):
        yield dict(zip(keys, instance))


@click.command()
@click.argument('train_files', nargs=-1, type=click.Path())
@click.option('--test_files', '-t', default=None, multiple=True, type=click.Path())
@click.option('--model_name', '-m', default=None, nargs=1, type=str)
@click.option('--exp', '-e', default=None, nargs=1, type=int)
@click.option('--kfolds', '-k', default=3, nargs=1, type=int)
def tune(train_files, test_files, model_name, exp, kfolds):
    # Complete args
    model_name = model_name.lower() if model_name else input('model_name=?').lower()
    exp_num = 'exp_'
    n = str(exp) if exp else input('exp_?')
    exp_num += n

    timestamp = get_timestamp()
    logger.log(f'** TUNING {timestamp} STARTS **')
    logger.info('---------- Cross validation {} on {} start ----------'.format(model_name, exp_num))

    # Load data
    logger.info('Loading training set: {}'.format(list(train_files)))
    train_df = load_corpus(list(train_files))
    if test_files != ():
        logger.info('Loading test set: {}'.format(list(test_files)))
        test_df = load_corpus(list(test_files))
    else:
        test_df = None

    # Load configs
    cfg_path = get_envar('CONFIG_PATH') + '/' + get_envar('BASE_CONFIG')
    logger.info('Loading base_configs from {}'.format(cfg_path))
    base_configs = read_config(cfg_path, obj_view=True)

    logger.info('Loading exp_configs on {} from {}'.format(exp_num, base_configs.exp_configs.path))
    exp_configs = read_config(base_configs.exp_configs.path, obj_view=False)[exp_num]


    tuning_params = exp_configs['hyperparams']
    for hyparams in product_dict(**tuning_params):
        description = exp_configs['description']
        logger.info('Experiment description: {}'.format(description.strip()))
        logger.info('Hyperparams: {}'.format(hyparams))
        logger.log(f'--- Running hyperparam {hyparams} ---')

        wdir = base_configs.model.savepath + get_timestamp() + '/'

        # CV
        kf = KFold(n_splits=kfolds, shuffle=True, random_state=42)

        cv = {}

        for k, (train_idx, val_idx) in enumerate(kf.split(train_df)):
            logger.info(f'-- Cross validation split {k+1} --')
            rec = train_validate(model_name, hyparams, train_df.iloc[train_idx], train_df.iloc[val_idx], test_df)
            cv.update({f'CV_{k+1}': rec})

        (_, cv_val, cv_test), df = best_scores(cv, complete=False)
        logger.info(f'**CV RESULTS** val_acc3={cv_val:.2%} test_acc3={cv_test:.2%}')
        # df.to_clipboard()
        logger.info(f'CV details \n{df}')
        logger.info('---------- Cross validation {} on {} end ----------'.format(model_name, exp_num))

        logger.log(f'CV details \n{df}')
        logger.log(f'**CV RESULTS** val_acc3={cv_val:.2%} test_acc3={cv_test:.2%}')
    logger.log(f'** TUNING {timestamp} ENDS **')

if __name__ == '__main__':
    tune()