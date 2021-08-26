import logging
from abc import ABCMeta

from .. import Classifier
from qiskit.compiler import transpile
from qiskit.circuit import QuantumRegister, QuantumCircuit
from qiskit.circuit import Qubit
from qiskit.transpiler import Layout
from qiskit.providers.basebackend import BaseBackend
from qiskit.visualization import plot_circuit_layout
from itertools import product
from typing import Union, Dict

logger = logging.getLogger(__name__)


class QuantumClassifier(Classifier, metaclass=ABCMeta):
    pass


class Qasvm_Mapping_4x2(Layout):
    """ Layout subclass to fed into QuantumInstance.initial_layout"""

    def __init__(self, backend: Union[str, BaseBackend, dict] = None, **qubits: dict):
        if backend is None or isinstance(backend, dict):
            super().__init__(backend)
        else:
            self.backend = backend
            self.registers = dict()
            self.registers['a'] = QuantumRegister(1, 'a')
            self.registers['i'] = QuantumRegister(2, 'i')
            self.registers['xi'] = QuantumRegister(1, 'xi')
            self.registers['yi'] = QuantumRegister(1, 'yi')
            self.registers['j'] = QuantumRegister(2, 'j')
            self.registers['xj'] = QuantumRegister(1, 'xj')
            self.registers['yj'] = QuantumRegister(1, 'yj')

            config = self.backend.configuration()
            if config.n_qubits < 9:
                raise QuantumError(f'At least 9 qubits required, but the backend has {config.n_qubits} qubits')
            if len(qubits) > 9:
                raise QuantumError(f'Specify 9 qubits instead of {len(qubits)}')
            try:
                second_dict = dict()
                second_dict[qubits['a']] = Qubit(self.registers['a'], 0)
                second_dict[qubits['i0']] = Qubit(self.registers['i'], 0)
                second_dict[qubits['i1']] = Qubit(self.registers['i'], 1)
                second_dict[qubits['xi']] = Qubit(self.registers['xi'], 0)
                second_dict[qubits['yi']] = Qubit(self.registers['yi'], 0)
                second_dict[qubits['j0']] = Qubit(self.registers['j'], 0)
                second_dict[qubits['j1']] = Qubit(self.registers['j'], 1)
                second_dict[qubits['xj']] = Qubit(self.registers['xj'], 0)
                second_dict[qubits['yj']] = Qubit(self.registers['yj'], 0)

                first_dict = dict()
                first_dict[qubits['a']] = Qubit(self.registers['a'], 0)
                first_dict[qubits['i0']] = Qubit(self.registers['i'], 0)
                first_dict[qubits['i1']] = Qubit(self.registers['i'], 1)
                first_dict[qubits['xi']] = Qubit(self.registers['xi'], 0)
                first_dict[qubits['yi']] = Qubit(self.registers['yi'], 0)
                first_dict[qubits['xj']] = Qubit(self.registers['xj'], 0)
            except KeyError as e:
                raise QuantumError(f"Qubit name '{e}' is missing.")
            self.second_dict = second_dict
            self.first_dict = first_dict
            super().__init__(self.second_dict)

    def plot(self, view: str = 'virtual'):
        qc = QuantumCircuit(*tuple(self.registers.values()))
        qc = transpile(qc, backend=self.backend, initial_layout=self)
        fig = plot_circuit_layout(qc, self.backend, view)
        return fig


def postprocess_Z_expectation(n: int, dic: Dict[str, float], *count):
    """ interpretation of qiskit result. a.k.a. parity of given qubits 'count' """
    temp = 0
    for binary in product((0, 1), repeat=n):
        val1 = (-1) ** sum([binary[c] for c in count])
        val2 = dic.get(''.join(map(str, binary)), 0)
        temp += val1 * val2
    return temp / sum(dic.values())


# errors
class QuantumError(BaseException):
    pass
