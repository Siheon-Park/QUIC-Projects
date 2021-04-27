
import logging

from .. import Classifier
from qiskit.circuit import QuantumRegister
from qiskit.circuit import Qubit
from qiskit.aqua import AquaError
from qiskit.transpiler import Layout
from qiskit.providers.basebackend import BaseBackend
from itertools import product
from typing import Union, Dict

logger = logging.getLogger(__name__)

class QuantumClassifier(Classifier):
    pass

class Qasvm_Mapping_4x2(Layout):
    """ order: a, i0, i1, xi, yi, j0, j1, xj, yj """

    def __init__(self, backend:Union[str, BaseBackend, dict]=None):
        if backend is None or isinstance(backend, dict):
            super().__init__(backend)
        else:
            self._backend_name = backend if isinstance(backend, str) else backend.name()
            self._QUBIT_LISTS = [Qubit(QuantumRegister(1, 'a'), 0),
                            Qubit(QuantumRegister(2, 'i'), 0),
                            Qubit(QuantumRegister(2, 'i'), 1),
                            Qubit(QuantumRegister(1, 'xi'), 0),
                            Qubit(QuantumRegister(1, 'yi'), 0),
                            Qubit(QuantumRegister(2, 'j'), 0),
                            Qubit(QuantumRegister(2, 'j'), 1),
                            Qubit(QuantumRegister(1, 'xj'), 0),
                            Qubit(QuantumRegister(1, 'yj'), 0)]
            if 'sydney' in self._backend_name:
                self.updated_date = '2021/04/19 01:12'
                self._prefered_mapping_order = [23, 15, 17, 21, 18, 26, 22, 24, 25]
            elif 'toronto' in self._backend_name:
                self.updated_date = '2021/04/28 01:00'
                self._prefered_mapping_order = [23, 15, 17, 21, 18, 22, 26, 24, 25]
            else:
                raise QuantumError('No support for {:}'.format(self._backend_name))
            super().__init__(dict(zip(self._QUBIT_LISTS, self._prefered_mapping_order)))
    
    @property
    def _layout_for_first_order_circuit(self):
        _vqs = self._QUBIT_LISTS[:5]+self._QUBIT_LISTS[7:8]
        _pqs = self._prefered_mapping_order[:5]+self._prefered_mapping_order[7:8]
        return Qasvm_Mapping_4x2(dict(zip(_vqs, _pqs)))

def postprocess_Z_expectation(n:int, dic:Dict[str, float], *count):
    ''' interpretaion of qiskit result. a.k.a. parity of given qubits 'count' '''
    temp = 0
    for bin in product((0,1), repeat=n):
        val1 = (-1)**sum([bin[c] for c in count])
        val2 = dic.get(''.join(map(str, bin)), 0)
        temp += val1*val2
    return temp/sum(dic.values())

# errors
class QuantumError(AquaError):
    pass

            