# ATLX
Implementation details of experiments described in paper  
[Attention and Lexicon Regularized LSTM for Aspect-based Sentiment Analysis](https://www.aclweb.org/anthology/P19-2035.pdf)

Experiments implemented with `python=3.6`, `tensorflow=1.5.0`, see [`requirements`](https://github.com/LingxB/atlx/blob/master/requirements.txt) for more details.

## Project Organization

    ├── configs                         -> Experiment configurations
    ├── data                            -> Datasets
        ├── ...                 
            ├── glove_lookup.parquet    -> Glove vectors of corpus vocabulary in compressed format
            ├── glove_symdict.yml       -> Corpus vocabulary
            ├── *.test.csv              -> Training set
            ├── *.train.csv             -> Test set
    ├── logs                            
    ├── models                          -> Directory for saving trained models
    ├── src                             
        ├── ...
            ├── models
                ├── atlstm.py           -> AT-LSTM model implementaion (Wang et al.)
                ├── atlx.py             -> ATLX model implementaion
    ...
       

## Getting started

**Build environment:**

    pip install -r requirements.txt
    
**Run experiments from project directory (Windows):**

AT-LSTM (baseline)

    script cross_validate data/processed/SemEval14/SemEval14_train.csv -t data/processed/SemEval14/SemEval14_test.csv -m atlx -e 1 -k 6
    
ATLX    
    
    script cross_validate data/processed/SemEval14/SemEval14_train.csv -t data/processed/SemEval14/SemEval14_test.csv -m atlx -e 3 -k 6

