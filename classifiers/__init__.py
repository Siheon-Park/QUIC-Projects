from abc import ABC, abstractmethod
import numpy as np
import logging
from sklearn.metrics import accuracy_score
from .datasets import DatasetError
from .utils import dill_save
from typing import Union, List

def process_info_from_alpha(alpha:np.ndarray, data:np.ndarray):
    support_ = np.argwhere(alpha>0.01*max(alpha)).flatten()
    support_vectors_ = data[support_]
    n_support_ = len(support_)
    return support_, support_vectors_, n_support_

class Classifier(ABC):
    def __init__(self, data:np.ndarray, label:np.ndarray):
        self.data = data
        self.label = label
        self.num_data = data.shape[0]
        self.dim_data = data.shape[1]
        self.alpha = None
        self.name = type(self).__name__
        if self.num_data != label.size:
            raise DatasetError('Not enough/More number of labels compare to dataset')

    @abstractmethod
    def f(self, testdata:Union[np.ndarray, List[np.ndarray]]):
        """ calculate optimal classifying value
            Args:
                testdata: data
            Return:
                f(np.ndarray): f(X)
        """
        raise NotImplementedError

    def predict(self, testdata:np.ndarray):
        """ predict label of data
            Args:
                testdata: data
            Return:
                predicted label(np.ndarray): y \in {0, 1}
        """
        return np.where(self.f(testdata)>0, 1., 0.)

    def accuracy(self, testdata:np.ndarray, testlabel:np.ndarray):
        """ calculate accuracy
            Args:
                testdata: data
                testlabel: y \in {0, 1}
            Return:
                accuracy(float)
        """
        return accuracy_score(self.predict(testdata), testlabel)
    
    def save(self, filepath):
        dill_save(self, filepath)
        logger = logging.getLogger(self.__module__)
        logger.info('{:} {:} is saved via package "dill" at {:}'.format(str(self.__module__), self.__class__.__name__, filepath))