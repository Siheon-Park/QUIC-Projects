import numpy as np
from qasvm.classifier import BinarySVM
from qiskit.circuit import QuantumCircuit, QuantumRegister, ClassicalRegister

class BinaryQASVM(BinarySVM):
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