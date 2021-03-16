""" Quantum State modification 
    Useage: state, n = quantum_state(classical_state)
"""

import math
import numpy as np
from typing import List, Optional, Any
from qiskit import QiskitError

_EPS = 1e-10


class quantum_state:
    """ return quantum version of state and number of qubit to represent state
        properties:
            state: semi-perfect statevector
            numOfqubits: number of qubits to repr self.state
            """

    def __init__(self, state):
        """state->statevector"""
        self.state = np.array(state)
        try:
            self._check_state(eps=_EPS)
        except QuantumStateNotVectorError:
            self.__init__(np.reshape(self.state, np.size(self.state)))
        except QuantumStateNotNormalError:
            self.__init__(self._normalize())
        except QuantumStateNotPowerOf2Error:
            self.__init__(self._appendzero())
        else:  # self.state is perfect qunautum state
            self.numOfqubits = int(math.log2(len(self.state)))

    def state_and_num(self):
        return [self.state, self.numOfqubits]

    def _check_state(self, eps=_EPS):
        """check if params is vaild size, normalized, and 1xD vector
            return number of qubit enought to represent params if nothing's wrong
            else, raise QiskitError """
        num_qubits = math.log2(len(self.state))
        if num_qubits == 0:
            raise QuantumStateJustScalarError(
                "Desired statevector is not vector")
        # Check if param is a power of 2
        if not num_qubits.is_integer():
            raise QuantumStateNotPowerOf2Error(
                "Desired statevector length not a positive power of 2.")
        # Check if probabilities (amplitudes squared) sum to 1
        if not self._is_normalized(eps=eps):
            raise QuantumStateNotNormalError(
                "Sum of amplitudes-squared does not equal one.")

        # Check if 1xD vector
        if not len(self.state) == np.size(self.state):
            raise QuantumStateNotVectorError(
                "Desired statevecter is not 1 by N vector")



    def _normalize(self, eps=_EPS):
        """ return normalized state """
        self.state = np.array(self.state)/np.linalg.norm(self.state)
        if not self._is_normalized(eps=eps):
            raise QuantumStateNotNormalError("Imposible to Normalize!!")
        else:
            return self.state

    def _is_normalized(self, eps=_EPS):
        """ test if vec is normalized enough 
            return Bool """
        # Check if probabilities (amplitudes squared) sum to 1
        if not math.isclose(sum(np.absolute(self.state) ** 2), 1.0, abs_tol=eps):
            #raise QiskitError("Sum of amplitudes-squared does not equal one.")
            return False
        else:
            return True

    def _appendzero(self):
        """ append zero until state size is power of 2 """
        n = len(self.state)
        N = 2**(math.ceil(math.log2(len(self.state))))
        new_state_R = np.zeros(N)
        new_state_I = np.zeros(N)
        new_state_R[:n] = np.real(self.state)
        new_state_I[:n] = np.imag(self.state)
        self.state = new_state_R + 1j * new_state_I
        return self.state

    def _numOfqubits(self):
        self._check_state()
        return int(math.log2(len(self.state)))
# error class


class QuantumStateNotNormalError(QiskitError):
    pass


class QuantumStateNotPowerOf2Error(QiskitError):
    pass


class QuantumStateNotVectorError(QiskitError):
    pass


class QuantumStateJustScalarError(QiskitError):
    pass


# Debug
if __name__ == '__main__':
    x1 = quantum_state([1, 1j, 2])
    x2 = quantum_state([1, -1j, 2])
    label = [1, 0]
    weight = quantum_state([1, 1])
    test = quantum_state([1, 0, 1])
