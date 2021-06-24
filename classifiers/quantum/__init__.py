import logging
from abc import ABCMeta

from .. import Classifier
from qiskit.circuit import QuantumRegister
from qiskit.circuit import Qubit
from qiskit.aqua import AquaError
from qiskit.transpiler import Layout
from qiskit.providers.basebackend import BaseBackend
from itertools import product
from typing import Union, Dict, List

logger = logging.getLogger(__name__)


class QuantumClassifier(Classifier, metaclass=ABCMeta):
    pass


class Qasvm_Mapping_4x2(Layout):
    """ Layout subclass to fed into QuantumInstance.initial_layout"""

    def __init__(self, backend: Union[str, BaseBackend, dict] = None, **qubits: dict):
        if backend is None or isinstance(backend, dict):
            super().__init__(backend)
        else:
            a = Qubit(QuantumRegister(1, 'a'), 0)
            i0 = Qubit(QuantumRegister(2, 'i'), 0)
            i1 = Qubit(QuantumRegister(2, 'i'), 1)
            xi = Qubit(QuantumRegister(1, 'xi'), 0)
            yi = Qubit(QuantumRegister(1, 'yi'), 0)
            j0 = Qubit(QuantumRegister(2, 'j'), 0)
            j1 = Qubit(QuantumRegister(2, 'j'), 1)
            xj = Qubit(QuantumRegister(1, 'xj'), 0)
            yj = Qubit(QuantumRegister(1, 'yj'), 0)
            config = backend.configuration()
            if config.n_qubits < 9:
                raise QuantumError(f'At least 9 qubits required, but the backend has {config.n_qubits} qubits')
            if len(qubits) > 9:
                raise QuantumError(f'Specify 9 qubits instead of {len(qubits)}')
            try:
                second_dict = dict()
                first_dict = dict()
                second_dict[qubits['a']] = a
                second_dict[qubits['i0']] = i0
                second_dict[qubits['i1']] = i1
                second_dict[qubits['xi']] = xi
                second_dict[qubits['yi']] = yi
                second_dict[qubits['j0']] = j0
                second_dict[qubits['j1']] = j1
                second_dict[qubits['xj']] = xj
                second_dict[qubits['yj']] = yj

                first_dict[qubits['a']] = a
                first_dict[qubits['i0']] = i0
                first_dict[qubits['i1']] = i1
                first_dict[qubits['xi']] = xi
                first_dict[qubits['yi']] = yi
                first_dict[qubits['xj']] = xj
            except KeyError as e:
                raise QuantumError(f"Qubit name '{e}' is missing.")
            self.second_dict = second_dict
            self.first_dict = first_dict
            super().__init__(self.second_dict)

    @property
    def _layout_for_first_order_circuit(self):
        return self.first_dict


def postprocess_Z_expectation(n: int, dic: Dict[str, float], *count):
    """ interpretation of qiskit result. a.k.a. parity of given qubits 'count' """
    temp = 0
    for binary in product((0, 1), repeat=n):
        val1 = (-1) ** sum([binary[c] for c in count])
        val2 = dic.get(''.join(map(str, binary)), 0)
        temp += val1 * val2
    return temp / sum(dic.values())


# errors
class QuantumError(AquaError):
    pass
