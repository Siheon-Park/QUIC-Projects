import numpy as np
from qiskit import QiskitError

class FeatureMap(object):
    pass

class PhaseFeatureMap(FeatureMap):
    def __init__(self)->None:
        pass

    def transform_data(self, data:np.ndarray) ->np.ndarray:
        if len(data.shape)==2:
            raise QiskitError(f'Shape of input data is {data.shape} (Expect 2D array)')
        N = data.shape[0]
        return np.array([np.exp(1j*x)/N for x in data])

    def circuit(self, data:np.ndarray):
        pass