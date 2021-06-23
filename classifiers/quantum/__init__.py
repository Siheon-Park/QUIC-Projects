from abc import ABCMeta

import numpy as np
import time
import logging

from qiskit.circuit.parametervector import ParameterVector
from .. import Classifier
from qiskit.circuit import QuantumRegister
from qiskit.circuit import Qubit
from qiskit.aqua import AquaError
from qiskit.aqua.algorithms import VQResult
from qiskit.aqua.components.optimizers import Optimizer
from qiskit.providers.basebackend import BaseBackend
from itertools import product
from typing import Union, Dict, Optional, Callable, Tuple, List

logger = logging.getLogger(__name__)


class QuantumClassifier(Classifier, metaclass=ABCMeta):
    @staticmethod
    def run_optimizer(
            optimizer: Optimizer,
            parameters: ParameterVector,
            cost_fn: Callable,
            initial_point: Optional[np.ndarray] = None,
            bounds: Optional[List[Tuple[float, float]]] = None,
            gradient_fn: Optional[Callable] = None) -> 'VQResult':

        num_parameters = len(parameters)

        if bounds is None:
            bounds = [(None, None)] * num_parameters

        if initial_point is not None and len(initial_point) != num_parameters:
            raise ValueError(
                'Initial point size {} and parameter size {} mismatch'.format(
                    len(initial_point), num_parameters))
        if len(bounds) != num_parameters:
            raise ValueError('Variational form bounds size does not match parameter size')
        # If *any* value is *equal* in bounds array to None then the problem does *not* have bounds
        problem_has_bounds = not np.any(np.equal(bounds, None))
        # Check capabilities of the optimizer
        if problem_has_bounds:
            if not optimizer.is_bounds_supported:
                raise ValueError('Problem has bounds but optimizer does not support bounds')
        else:
            if optimizer.is_bounds_required:
                raise ValueError('Problem does not have bounds but optimizer requires bounds')
        if initial_point is not None:
            if not optimizer.is_initial_point_supported:
                raise ValueError('Optimizer does not support initial point')
        else:
            if optimizer.is_initial_point_required:
                if initial_point is None:  # If still None use a random generated point
                    low = [(l if l is not None else -2 * np.pi) for (l, u) in bounds]
                    high = [(u if u is not None else 2 * np.pi) for (l, u) in bounds]
                    initial_point = np.random.uniform(low, high)

        start = time.time()
        if not optimizer.is_gradient_supported:  # ignore the passed gradient function
            gradient_fn = None
        else:
            if not gradient_fn:
                gradient_fn = None

        logger.info('Starting optimizer.\nbounds=%s\ninitial point=%s', bounds, initial_point)
        if num_parameters > 0:
            opt_params, opt_val, num_optimizer_evaluations = optimizer.optimize(num_parameters,
                                                                                cost_fn,
                                                                                variable_bounds=bounds,
                                                                                initial_point=initial_point,
                                                                                gradient_function=gradient_fn)
        else:
            opt_params, opt_val, num_optimizer_evaluations = ([], cost_fn([]), 0)
        eval_time = time.time() - start

        result = VQResult()
        result.optimizer_evals = num_optimizer_evaluations
        result.optimizer_time = eval_time
        result.optimal_value = opt_val
        result.optimal_point = opt_params
        result.optimal_parameters = dict(zip(parameters, opt_params))

        return result


class Qasvm_Mapping_4x2(object):
    """ order: a, i0, i1, xi, yi, j0, j1, xj, yj """

    def __init__(self, backend: Union[str, BaseBackend]):
        self.backend_name = backend if isinstance(backend, str) else backend.name()
        self.updated_date = None

    @property
    def layout(self):
        QUBIT_LISTS = [Qubit(QuantumRegister(1, 'a'), 0),
                       Qubit(QuantumRegister(2, 'i'), 0),
                       Qubit(QuantumRegister(2, 'i'), 1),
                       Qubit(QuantumRegister(1, 'xi'), 0),
                       Qubit(QuantumRegister(1, 'yi'), 0),
                       Qubit(QuantumRegister(2, 'j'), 0),
                       Qubit(QuantumRegister(2, 'j'), 1),
                       Qubit(QuantumRegister(1, 'xj'), 0),
                       Qubit(QuantumRegister(1, 'yj'), 0)]
        if 'sydney' in self.backend_name:
            self.updated_date = '2021/03/28 02:41'
            preferred_mapping_order = [14, 5, 9, 11, 8, 20, 22, 16, 19]
        elif 'toronto' in self.backend_name:
            self.updated_date = '2021/06/23 17:16'
            preferred_mapping_order = [12, 4, 6, 10, 7, 11, 16, 13, 14]
        else:
            raise QuantumError('No support for {:}'.format(self.backend_name))
        return {QUBIT_LISTS[i]: preferred_mapping_order[i] for i in range(len(QUBIT_LISTS))}


def postprocess_Z_expectation(n: int, dic: Dict[str, float], *count):
    """ interpretation of qiskit result. a.k.a. parity of given qubits 'count' """
    temp = 0
    for b in product((0, 1), repeat=n):
        val1 = (-1) ** sum([b[c] for c in count])
        val2 = dic.get(''.join(map(str, b)), 0)
        temp += val1 * val2
    return temp / sum(dic.values())


# errors
class QuantumError(AquaError):
    pass
