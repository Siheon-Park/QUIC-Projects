from matplotlib import pyplot as plt
import numpy as np
from itertools import product
from numpy.lib.arraysetops import isin
from sklearn.manifold import TSNE
import dill
from typing import Union, Optional, Dict, List, Callable
from qiskit.aqua.components.optimizers import Optimizer
import logging
import importlib
import pathlib
import uuid
import time

DEFAULT_EXP_LOG_PATH = pathlib.Path.home()/'QUIC-Projects'/'exp_logs'

def postprocess_Z_expectation(n:int, dic:Dict[str, int], *count):
    ''' interpretaion of qiskit result. a.k.a. parity of given qubits 'count' '''
    temp = 0
    for bin in product((0,1), repeat=n):
        val1 = (-1)**sum([bin[c] for c in count])
        val2 = dic.get(''.join(map(str, bin)), 0)
        temp += val1*val2
    return temp/sum(dic.values())

def tsne(data, perp:float=30, seed:int=None):
    ''' tsne '''
    np.random.seed(seed)
    return TSNE(n_components=2, perplexity=perp).fit_transform(data)

def get_optimizer_logger(optimizer:Optimizer, level:int=logging.DEBUG, handle:logging.Handler=None, form:logging.Formatter=None, **kwargs):
    ''' get logger defined in optimizer module and add handle with formatter and level '''
    logger = logging.getLogger(optimizer.__module__)
    if not isinstance(handle, logging.Handler):
        filepath = kwargs.get('filename', DEFAULT_EXP_LOG_PATH / (time.strftime('%y%m%d-%H%M%S-', time.localtime(time.time()))+str(uuid.uuid4())+'.log'))
        handle = logging.FileHandler(filepath)
    if not isinstance(form, logging.Formatter):
        form = logging.Formatter(kwargs.get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    handle.setFormatter(form)
    logger.addHandler(handle)
    logger.setLevel(level)
    return logger
        
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
    