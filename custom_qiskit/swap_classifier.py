import numpy as np

from qiskit.circuit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit import QiskitError
from qiskit import execute, Aer

from .entangler import Entangler
from .classifier import SVM, InvalidDataError

class _SWAP_classifier(SVM):
    """ swap classifier parent class """
    def __init__(self, data:np.ndarray, label:np.ndarray):
        super().__init__(data, label, None)
        self.index_qubit_num = int(np.ceil(np.log2(self.num_data)))
        self.data_qubit_num = int(np.ceil(np.log2(self.dim_data)))
        self.theta = None

    def create_entangling_circuit(self, circuit_id:int, num_layer:int, qr:QuantumRegister=None, pname:str='theta'):
        """ ref: https://arxiv.org/abs/1905.10876 """
        if not isinstance(qr, QuantumRegister):
            qr = QuantumRegister(self.index_qubit_num) 
        qc = Entangler(qr, circuit_id, num_layer, pname)
        return qc

    def _bind_parameter_return_alpha(self, params:np.ndarray, qc:QuantumCircuit, theta):
        if theta is not None:
            bind_qc = qc.bind_parameters({theta:np.pi*params})
        else:
            bind_qc = qc
        backend = Aer.get_backend('unitary_simulator')
        _matrix = execute(bind_qc, backend=backend).result().get_unitary()
        init_state = np.zeros(2**self.index_qubit_num)
        init_state[0] = 1.0
        alpha = np.abs(_matrix @ init_state)**2
        if not len(alpha) == self.num_data:
            raise InvalidDataError('number of data is not power of 2: may leak alpha')
        return alpha

    def _create_double_qc(self, id, layer):
        ''' dubble qc '''
        qr_a = QuantumRegister(1, name='ancila')
        qr_i = QuantumRegister(self.index_qubit_num, name='index_i')
        qr_xi = QuantumRegister(self.data_qubit_num, name='data_i')
        qr_yi = QuantumRegister(1, name='label_i')
        qr_j = QuantumRegister(self.index_qubit_num, name='index_j')
        qr_xj = QuantumRegister(self.data_qubit_num, name='data_j')
        qr_yj = QuantumRegister(1, name='label_j')
        cr = ClassicalRegister(3, name='c')
        self.qreg = [qr_a, qr_i, qr_xi, qr_yi, qr_j, qr_xj, qr_yj]
        self.creg = [cr]
        self.qc = QuantumCircuit(*(self.qreg+self.creg), name='SWAP classifier') # empty circuit
        self.weight_qc1 = self.create_entangling_circuit(id, layer, qr_i, 'theta_i') # weight encoding
        self.weight_qc2 = self.create_entangling_circuit(id, layer, qr_j, 'theta_j') # weight encoding
        self.theta1 = self.weight_qc1.theta
        self.theta2 = self.weight_qc2.theta
        self.qc = self.qc.combine(self.weight_qc1).combine(self.weight_qc2)
        [self.qc.ctrl_encode(self.data[i], i, qr_xi, qr_i, name=f"Data_i {i}") for i in range(self.num_data)] # data encoding
        [self.qc.ctrl_x(i, qr_yi, qr_i) if self.label[i]<0 else None for i in range(self.num_data)] # label encoding
        [self.qc.ctrl_encode(self.data[j], j, qr_xj, qr_j, name=f"Data_j {j}") for j in range(self.num_data)] # data encoding
        [self.qc.ctrl_x(j, qr_yj, qr_j) if self.label[j]<0 else None for j in range(self.num_data)] # label encoding
        self.qc.barrier()
        # SWAP test
        self.qc.h(qr_a)
        [self.qc.cswap(qr_a, qr_xi[k], qr_xj[k]) for k in range(self.data_qubit_num)]
        self.qc.h(qr_a)
        # measure
        self.qc.measure(qr_a, cr[1]) # pylint: disable=no-member
        self.qc.measure(qr_yi, cr[0]) # pylint: disable=no-member
        self.qc.measure(qr_yj, cr[2]) # pylint: disable=no-member

    def _create_single_qc(self, id:int, layer:int):
        ''' single qc '''
        qr_a = QuantumRegister(1, name='ancila')
        qr_i = QuantumRegister(self.index_qubit_num, name='index')
        qr_xi = QuantumRegister(self.data_qubit_num, name='data')
        qr_yi = QuantumRegister(1, name='label')
        qr_xj = QuantumRegister(self.data_qubit_num, name='testdata')
        cr = ClassicalRegister(3, name='c')
        self.class_qreg = [qr_a, qr_i, qr_xi, qr_yi, qr_xj]
        self.class_creg = [cr]
        self.class_qc = QuantumCircuit(*(self.class_qreg+self.class_creg), name='SWAP classifier') # empty circuit
        self.class_weight_qc = self.create_entangling_circuit(id, layer, qr_i, 'theta') # weight encoding
        self.class_theta = self.class_weight_qc.theta
        self.class_qc = self.class_qc.combine(self.class_weight_qc)
        [self.class_qc.ctrl_encode(self.data[i], i, qr_xi, qr_i, name=f"Data_{i}") for i in range(self.num_data)] # data encoding
        [self.class_qc.ctrl_x(i, qr_yi, qr_i) if self.label[i]<0 else None for i in range(self.num_data)] # label encoding
        self.class_qc.barrier()
        # SWAP test
        self.class_qc.h(qr_a)
        [self.class_qc.cswap(qr_a, qr_xi[k], qr_xj[k]) for k in range(self.data_qubit_num)]
        self.class_qc.h(qr_a)
        # measure
        self.class_qc.measure(qr_a, cr[1]) # pylint: disable=no-member
        self.class_qc.measure(qr_yi, cr[0]) # pylint: disable=no-member     

