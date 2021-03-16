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
        self.testerr = None

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


class Optimizer:
    def __init__(self, name:str):
        self.name = name

class NotSuitableClsOptPairError(BaseException):
    pass
class ClassifierError(BaseException):
    pass
class OptimizerError(BaseException):
    pass