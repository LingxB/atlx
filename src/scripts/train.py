from src.models import VALID_MODELS
from src.utils import Logger, __fn__, load_corpus, get_envar, read_config, get_timestamp
from src.data import AbsaDataManager, LexiconManager
import click



logger = Logger(__fn__())


@click.command()
@click.argument('train_files', nargs=-1, type=click.Path())
@click.option('--val_files', '-v', default=None, multiple=True, type=click.Path())
@click.option('--test_files', '-t', default=None, multiple=True, type=click.Path())
@click.option('--model_name', '-m', default=None, nargs=1, type=str)
@click.option('--exp', '-e', default=None, nargs=1, type=int)
def train(train_files, val_files, test_files, model_name, exp):
    # Complete args
    model_name = model_name.lower() if model_name else input('model_name=?').lower()
    exp_num = 'exp_'
    n = str(exp) if exp else input('exp_?')
    exp_num += n

    logger.info('---------- Training {} on {} start ----------'.format(model_name, exp_num))

    # Load data
    logger.info('Loading training set: {}'.format(list(train_files)))
    train_df = load_corpus(list(train_files))
    if val_files != ():
        logger.info('Loading validation set: {}'.format(list(val_files)))
        val_df = load_corpus(list(val_files))
    else:
        val_df = None
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

    description = exp_configs['description']
    hyparams = exp_configs['hyperparams']
    logger.info('Experiment description: {}'.format(description.strip()))
    logger.info('Hyperparams: {}'.format(hyparams))

    wdir = base_configs.model.savepath + get_timestamp() + '/'

    # Build Model
    lm = LexiconManager()
    dm = AbsaDataManager(lexicon_manager=lm)
    model = VALID_MODELS[model_name.lower()]
    model = model(datamanager=dm, parameters=hyparams)

    # Train
    model.train(train_df, val_df, test_df)

    # Predict and score on test
    if test_df is not None:
        _, _, loss_, acc3_ = model.score(test_df)
        logger.info('Final score on test set: '
                    'test_loss={loss:.4f} ' \
                    'test_acc3={acc:.2%}'\
                    .format(loss=loss_, acc=acc3_))

    # Save model
    model.save(wdir)

    # Close tf.Session, not really necessary but... anyway
    model.close_session()

    logger.info('---------- Training {} on {} end ----------'.format(model_name, exp_num))




if __name__ == '__main__':
    train()





















# -------------------------------------------

#
# base_configs = read_config(get_envar('CONFIG_PATH') + '/' + get_envar('BASE_CONFIG'), obj_view=True)
# exp_num = 'exp_' + '4'
# exp_configs = read_config(base_configs.exp_configs.path, obj_view=False)[exp_num]
# hyparams = exp_configs['hyperparams']
# description = exp_configs['description']
# wdir = base_configs.model.path + get_timestamp() + '/'
#
#
# lm = LexiconManager()
# dm = AbsaDataManager(lexicon_manager=lm)
#
# train_df = load_corpus('data/processed/ATAE-LSTM/train.csv')
# dev_df = load_corpus('data/processed/ATAE-LSTM/dev.csv')
# test_df = load_corpus('data/processed/ATAE-LSTM/test.csv')
#
# hyparams['epochs'] = 2
#
# model = ATLSTM(datamanager=dm, parameters=hyparams)
#
#
# model.train(train_df, test_df)
#
# model.predict(dev_df)
#
#
# model.save(wdir)




