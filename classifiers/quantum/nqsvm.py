import logging
import numpy as np
from qiskit.circuit.parametervector import ParameterVector
from qiskit.circuit import QuantumCircuit
from qiskit.utils import QuantumInstance
from sympy.logic.boolalg import Boolean
from . import QuantumError
from . import QuantumClassifier
from typing import Any, Union, Optional, Dict, List, Iterable

from qiskit_machine_learning.kernels import QuantumKernel
from qiskit.algorithms.optimizers import spsa

logger = logging.getLogger(__name__)


class NormQSVM(QuantumClassifier):
    def __init__(self, data: np.ndarray, label: np.ndarray,
                 quantum_instance: QuantumInstance, lamda: float = 1.0,
                 feature_map: QuantumCircuit = None, var_form: QuantumCircuit = None,
                 initial_point: np.ndarray = None):
        super().__init__(data, label)
        del self.alpha
        self.polary = 2 * self.label - 1
        self.quantum_instance = quantum_instance
        self._qk = QuantumKernel(feature_map=feature_map, quantum_instance=quantum_instance, enforce_psd=False)
        self.kernel_matrix = np.abs(self._qk.evaluate(self.data, self.data)) ** 2
        self.feature_map = feature_map
        self.var_form = var_form
        self.lamda = lamda

        if self.var_form is None:
            self.parameters = OptParameters(ParameterVector('theta', self.num_data))
        else:
            self.parameters = OptParameters(self.var_form.parameters)
        self.num_parameters = len(self.parameters)

        if initial_point is None:
            self.initial_point = np.pi * (2 * np.random.random(self.num_parameters) - 1)
        else:
            self.initial_point = initial_point
        self.parameters.update(self.initial_point)

    def alpha(self, params: np.ndarray):
        if self.var_form is None:
            return params / sum(np.abs(params))
        else:
            var_qc = self.var_form.assign_parameters(dict(zip(self.parameters.param_tag, params)))
            var_qc.save_statevector()
            result = self.quantum_instance.execute(var_qc)
            return np.abs(result.get_statevector()) ** 2

    def cost_fn(self, params: np.ndarray):
        alpha = self.alpha(params)
        beta = alpha * self.polary
        K = self.kernel_matrix
        ret = beta @ K @ beta.reshape(-1, 1)
        return ret.item()

    def f(self, testdata):
        beta = self.alpha(self.parameters) * self.polary
        K = np.abs(self._qk.evaluate(self.data, testdata)) ** 2
        return beta @ K

    def accuracy(self, testdata, testlabel):
        yt = 2 * testlabel - 1
        return sum(yt * self.f(testdata) > 0) / len(yt)


class OptParameters(np.ndarray):
    def __new__(cls, params: ParameterVector, *args, **kwargs):
        shape = len(params)
        obj = super().__new__(cls, *args, shape=shape, **kwargs)
        obj.param_tag = params
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self.param_tag = getattr(obj, 'param_tag', None)

    def to_dict(self):
        return dict(zip(self.param_tag, self))

    def update(self, params: Union[List, np.ndarray, Iterable]):
        for i, p in enumerate(params):
            self[i] = p

    def __repr__(self):
        return self.to_dict().__repr__()
