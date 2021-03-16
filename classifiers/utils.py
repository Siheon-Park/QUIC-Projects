from matplotlib import pyplot as plt
import numpy as np
from itertools import product
from numpy.lib.arraysetops import isin
from sklearn.manifold import TSNE
import logging
import pathlib
import uuid
import time
import dill
import pickle
import re

DEFAULT_EXP_LOG_PATH = pathlib.Path.home()/'QUIC-Projects'/'exp_logs'

def tsne(data, perp:float=30, seed:int=None):
    ''' tsne '''
    np.random.seed(seed)
    return TSNE(n_components=2, perplexity=perp).fit_transform(data)

def get_loggers_add_handle(*objects, level:int=logging.DEBUG, handle:logging.Handler=None, form:logging.Formatter=None, **kwargs):
    ''' get logger defined in module and add handle with formatter and level '''
    loggers = []
    for obj in objects:
        logger = logging.getLogger(obj.__module__)
        if not isinstance(handle, logging.Handler):
            filepath = kwargs.get('filename', DEFAULT_EXP_LOG_PATH / (time.strftime('%y%m%d-%H%M%S-', time.localtime(time.time()))+str(uuid.uuid4())+'.log'))
            handle = logging.FileHandler(filepath)
        if not isinstance(form, logging.Formatter):
            form = logging.Formatter(kwargs.get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        handle.setFormatter(form)
        logger.addHandler(handle)
        logger.setLevel(level)
        loggers.append(logger)
    return loggers
        
def clean_exp_logs(dir_path:pathlib.Path=None):
    """ remove files with extensions .log, .dill if is not pared (i.e. a.dill-a.log)"""
    if dir_path is None:
        dir_path = DEFAULT_EXP_LOG_PATH
    dill_files = list(dir_path.glob('*.dill'))
    log_files = list(dir_path.glob('*.log'))
    dill_names = list(map(lambda x:x.stem, dill_files))
    log_names = list(map(lambda x:x.stem, log_files))
    for name, p in zip(log_names, log_files):
        if name not in dill_names:
            print(f'rm {p}')
            p.unlink()
    for name, p in zip(dill_names, dill_files):
        if name not in log_names:
            print(f'rm {p}')
            p.unlink()

def dill_save(self, filepath):
    with open(filepath, 'wb') as f:
        dill.dump(self, f)

def load_from_log(log_file:pathlib.Path, class_name:str, prefix:str=' is saved via package "dill" at '):
    with open(log_file, 'r') as f:
        lines = f.readlines()
    lines.reverse()
    letter = re.compile('.*'+class_name+prefix+'.*')
    for l in lines:
        found_letter = letter.search(l)
        if found_letter is not None:
            dill_file = found_letter.group().split(class_name+prefix)[-1]
            with open(dill_file, 'rb') as f:
                obj = dill.load(f)
            return obj
    raise FileNotFoundError('"~.dill" is not in {:}'.format(log_file))

    