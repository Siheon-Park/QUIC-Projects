import numpy as np
import logging
from .datasets import DatasetError
from .utils import dill_save

def process_info_from_alpha(alpha:np.ndarray, data:np.ndarray):
    support_ = np.argwhere(alpha>0.01*max(alpha)).flatten()
    support_vectors_ = data[support_]
    n_support_ = len(support_)
    return support_, support_vectors_, n_support_

class Classifier:
    def __init__(self, data:np.ndarray, label:np.ndarray):
        self.data = data
        self.label = label
        self.num_data = data.shape[0]
        self.dim_data = data.shape[1]
        self.alpha = None
        self.name = type(self).__name__
        if self.num_data != label.size:
            raise DatasetError('Not enough/More number of labels compare to dataset')
    
    def save(self, filepath):
        dill_save(self, filepath)
        logger = logging.getLogger(self.__module__)
        logger.info('{:} {:} is saved via package "dill" at {:}'.format(str(self.__module__), self.__class__.__name__, filepath))