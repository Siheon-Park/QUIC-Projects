
from qiskit.circuit import QuantumRegister
from qiskit.circuit import Qubit
from qiskit.aqua import AquaError
from qiskit.providers.basebackend import BaseBackend
from itertools import product
from typing import Union, Dict


class Qasvm_Mapping_4x2(object):
    """ order: a, i0, i1, xi, yi, j0, j1, xj, yj """

    def __init__(self, backend:Union[str, BaseBackend]):
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
            self.updated_date = '2021/03/14 18:55'
            prefered_mapping_order = [3, 0, 4, 2, 1, 9, 11, 5, 8]
        elif 'toronto' in self.backend_name:
            self.updated_date = '2021/03/14 18:55'
            prefered_mapping_order = [3, 0, 4, 2, 1, 9, 11, 5, 8]
        else:
            raise QuantumError('No support for {:}'.format(self.backend_name))
        return {QUBIT_LISTS[i]:prefered_mapping_order[i] for i in range(len(QUBIT_LISTS))}

def postprocess_Z_expectation(n:int, dic:Dict[str, int], *count):
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

            