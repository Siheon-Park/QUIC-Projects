import numpy as np
from typing import List, Any
import abc

class Data(metaclass=abc.ABCMeta):
    def __init__(self, dim:int, num:int, params:Any):
        self.dim = dim
        self.num = num
        self.params = params
    
    @abc.abstractmethod
    def funcname(self, parameter_list):
        raise NotImplementedError()

class RandomData(Data):
    def __init__(self, dim:int, num:int, params: Any):
        super().__init__(dim, num, params)
    def funcname(self):
        return 'asdf'

class RandomQuantumData(RandomData):
    pass

class RandomClassicalData(RandomData):
    pass
    
if __name__ == '__main__':
    a = RandomData(2, 2, 'asdf')
    print(a)