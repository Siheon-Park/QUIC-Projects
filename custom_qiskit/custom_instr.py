import math
import numpy as np
from typing import List, Optional, Any, Dict
from qiskit.exceptions import QiskitError
from qiskit.circuit import QuantumCircuit
from qiskit.circuit import QuantumRegister
from qiskit.circuit import ClassicalRegister
from qiskit.circuit import Instruction, Gate
from qiskit.circuit.library.standard_gates.x import XGate
from qiskit import transpile

from prepare_state import quantum_state
import quantum_encoder



_EPS = 1e-10  # global variable used to chop very small numbers to zero

# Classifiter circuit template


class Swap_classifier(Instruction):
    def __init__(self, weight:List, dataset:List[List], label:List, testdata:List):
        """
            Create SWAP test classification composite.

            params (list): list of vectors
            params[0] (ListLike): weight vector
            params[1] (ListLike): array of data
                                x = array([...x1...], 
                                            [...x2...], 
                                            ...)
            params[2] (ListLike): list of labels of data
            params[3] (ListLike): list of test data
        """
        # weight = parmas[0] # len: number of data
        # dataset = params[1] # len: number of data
        # label = params[2] # len: number of data
        # testdata = params[3] # len: dimention of data
        self.original_params = [weight, dataset, label, testdata]
        # modify params to valid quanutm state
        weight = quantum_state(weight)
        # label: binary array or sign array
        testdata = quantum_state(testdata)
        dataset = np.array([quantum_state(data).state_and_num() for data in dataset]).transpose()
        index_qubit_num = weight.numOfqubits
        data_qubit_num = testdata.numOfqubits

        # check dataset & test data
        if not np.all(dataset[1] == data_qubit_num):
            raise DataPreparationError("invalid data set or test data")
        if not len(label)==len(dataset[0]):
            raise DataPreparationError("invalid label")
        qubit_numbers = [1, index_qubit_num, data_qubit_num, 1, data_qubit_num]
        num_qubits = int(sum(qubit_numbers))
        params = [weight.state, dataset[0,:], label, testdata.state, qubit_numbers]
        super().__init__("SWAP classifier", num_qubits, 2, params)
        self.number_of_data = len(dataset[0])
        self.weight= weight.state
        self.dataset=dataset[0]
        self.label= label
        self.testdata= testdata.state
        self.index_qubit_num= index_qubit_num
        self.data_qubit_num= data_qubit_num

    def _define(self):
        """
            params (list): list of vectors
            params[0] (ListLike): weight vector
            params[1] (ListLike): array of data
                                x = array([...x1...], 
                                            [...x2...], 
                                            ...)
            params[2] (ListLike): list of labels of data
            params[3] (ListLike): list of test data
        """
        number_of_data = self.number_of_data
        weight= self.weight
        dataset=self.dataset
        label= self.label
        testdata= self.testdata
        index_qubit_num= self.index_qubit_num
        data_qubit_num= self.data_qubit_num

        qr_a = QuantumRegister(1, name='ancila')
        qr_i = QuantumRegister(index_qubit_num, name='index')
        qr_x = QuantumRegister(data_qubit_num, name='data')
        qr_y = QuantumRegister(1, name='label')
        qr_xt = QuantumRegister(data_qubit_num, name='test')
        cr = ClassicalRegister(2, name='c')

        qc = QuantumCircuit(qr_a, qr_i, qr_x, qr_y, qr_xt, cr, name='SWAP classifier')
        # weight and test data encoding
        qc.encode(weight, qr_i, name='Weight')
        qc.encode(testdata, qr_xt, name='Test Data')
        qc.barrier()
        
        # data and label, index encoding
        [qc.ctrl_encode(dataset[i], i, qr_x, qr_i, name=f"Data {i}") for i in range(number_of_data)]
        [qc.ctrl_x(i, qr_y, qr_i) if label[i]>0 else None for i in range(number_of_data)]
        qc.barrier()
        # SWAP test
        qc.h(qr_a)
        [qc.cswap(qr_a, qr_x[i], qr_xt[i]) for i in range(data_qubit_num)]
        qc.h(qr_a)
        # measure
        qc.measure(qr_a, cr[0]) # pylint: disable=no-member
        qc.measure(qr_y, cr[1]) # pylint: disable=no-member
        self.definition = qc

    def circuit(self):
        return self.definition
    
def process(result, classtype=Swap_classifier):
    if classtype is Swap_classifier:
        c00 = result.get('00', 0)
        c01 = result.get('01', 0)
        c10 = result.get('10', 0)
        c11 = result.get('11', 0)
        return (c00 + c11 - c01 - c10) / sum(result.values())

class QuadraticOptimizer



#-------------------------------------------------------------------------------
class DataPreparationError(QiskitError):
    pass

if __name__ == '__main__':
    # toy test
    # toy test
    x1 = [1, 1j, 0]
    x2 = [1, -1j, 0]
    x3 = [1, 1, 1]
    label = [1, 0, 1]
    weight = [1,1,1]
    test = [1,0, 0]

    classifier = Swap_classifier(weight, [x1, x2, x3], label, test)
    swap_qc = classifier.circuit()
    swap_qc.draw('mpl')