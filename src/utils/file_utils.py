import yaml
from dotenv import load_dotenv, find_dotenv
import os
import inspect
from src.utils.att_dict import AttributeDict
from src.utils.data_utils import create_symbol_dict
import warnings
import pandas as pd
from time import time
from datetime import datetime
import pickle

def get_timestamp():
    ts = time()
    ts = datetime.fromtimestamp(ts).strftime('%Y%m%d_%H%M%S')
    return ts

def read_config(config_file, obj_view=True):
    cfg = read_yaml(config_file + '.yml')
    if obj_view:
        cfg = AttributeDict(**cfg)
    return cfg

def save_yaml(dict, path, mode='w'):
    with open(path, mode) as outfile:
        yaml.dump(dict, outfile, default_flow_style=False)

def read_yaml(yaml_file):
    with open(yaml_file, 'r') as ymlfile:
        yml = yaml.load(ymlfile)
    return yml

def get_envar(key):
    """Get environment variable form .env file"""
    load_dotenv(find_dotenv(raise_error_if_not_found=True, usecwd=True))
    return os.environ.get(key)

def list_files(path):
    l = list(os.walk(path))
    if l == []:
        return l
    else:
        return l[0][-1]

def load_symbod_dict_if_exists(load_path):
    warnings.warn('Function is deprecated!')
    try:
        sd = read_yaml(load_path)
    except FileNotFoundError:
        sd = False
    return sd

def create_dump_symbol_dict(corpus, start_idx, dump_path):
    warnings.warn('Function is deprecated!')
    sd, _ = create_symbol_dict(corpus, start_idx)
    save_yaml(sd, dump_path, 'x')
    return sd

def __fn__():
    """
    Magic function to return the python file/module name where the function is called. If called in console, return
    ``'__main__'``.

    Returns
    -------
    str
        File/module name where the function is called, ``'__main__'`` when in console.
    """
    frame = inspect.stack()[1]
    module = inspect.getmodule(frame[0])
    try:
        name = os.path.basename(module.__file__)[:-3]
    except AttributeError:
        name = '__main__'
    return name

def load_corpus(file, **kwargs):
    if isinstance(file, str):
        _df = pd.read_csv(file, **kwargs)
    elif isinstance(file, list):
        _df = pd.concat([pd.read_csv(f, **kwargs) for f in file], ignore_index=True)
    else:
        raise AttributeError('File type not valid, path or list of paths only.')
    return _df

def load_embedding(file, **kwargs):
    df = pd.read_parquet(file, **kwargs)
    return df

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def pickle_dump(to_dump, path):
    with open(path, 'wb') as handle:
        pickle.dump(to_dump, handle, protocol=pickle.HIGHEST_PROTOCOL)

def pickle_load(path):
    with open(path, 'rb') as handle:
        return pickle.load(handle)