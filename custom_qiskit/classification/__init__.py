import numpy as np
import math
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod

class Classifier(ABC):
    def __init__(self, data:np.ndarray, label:np.ndarray, name:str):
        """
            data = np.array(num_data, dim_data)
            label = np.array
            name = str

            @property: data, label, is_opt, name
            should impliment: optimize, classifiy
        """
        super().__init__()
        self._data = data
        self._label = label
        self._name=name
        self._is_opt = False
        self._opt_dict = None

    @property
    def data(self):
        return self._data

    @property 
    def label(self):
        return self._label

    @property
    def is_opt(self):
        return self._is_opt

    @property
    def name(self):
        return self._name

    @property
    def opt_dict(self):
        return self._opt_dict

    @abstractmethod
    def optimize(self, **kwargs):
        pass
    
    @abstractmethod
    def classify(self, test, **kwargs):
        pass
    
    def check_perfomance(self, test_data:np.ndarray, test_label: np.ndarray):
        _temp = self.classify(test_data)==test_label
        performance = np.sum(_temp)/_temp.size
        return performance


class Optimizer:
    def __init__(self, name:str):
        self.name = name
        self.opt_variable = None
        self.opt_result = None

class NotSuitableClsOptPairError(BaseException):
    pass