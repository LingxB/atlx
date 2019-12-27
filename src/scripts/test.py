from src.models import VALID_MODELS
from src.utils import Logger, __fn__, load_corpus, list_files, get_timestamp
import click
import numpy as np


logger = Logger(__fn__())


@click.command()
@click.argument('model_dir', nargs=1, type=click.Path())
@click.argument('test_files', nargs=-1, type=click.Path())
@click.option('--out_file', '-o', default='data/score', type=click.Path())
def test(model_dir, test_files, out_file):

    # model_dir = 'models/atlx/SemEval14/baseline'
    # test_files = ['data/processed/SemEval14/SemEval14_test.csv']

    # Get model name
    model_files = list_files(model_dir)
    model_name = list(set([n.split('.')[0] for n in model_files if n.split('.')[-1] in {'index','meta'}]))
    assert len(model_name) == 1, 'Multiple model names in model directory!'
    model_name = model_name[0]
    logger.info('Found model {} in given directory'.format(model_name.lower()))

    # Load model
    model = VALID_MODELS[model_name.lower()]
    model = model()
    model.load(model_dir)

    # Load data
    test_df = load_corpus(list(test_files))

    # Score
    pred_, alpha_, loss_, acc3_ = model.score(test_df)
    logger.info('test_loss={loss:.4f} ' \
                'test_acc3={acc:.2%}'\
                .format(loss=loss_, acc=acc3_))


    prediction = np.argmax(pred_, axis=1)
    label_dict = {0: -1,
                  1: 0,
                  2: 1
                  }
    test_df['PRED'] = [label_dict[p] for p in prediction]
    test_df['NEG'], test_df['NEU'], test_df['POS'] = pred_[:,0], pred_[:,1], pred_[:,2]
    test_df['ALPHA'] = np.squeeze(alpha_).tolist()

    logger.info(f'Recomputing in output dataframe: test_acc3={(test_df.CLS == test_df.PRED).value_counts()[True]/len(test_df):.2%}')

    wdir = f'{out_file}/pred_{get_timestamp()}.csv'
    test_df.to_csv(wdir, index=False)
    logger.info(f'Output saved to {wdir}')

if __name__ == '__main__':
    test()