
import numpy as np
import math
from qiskit.circuit.parameter import Parameter
from qiskit.circuit.parametervector import ParameterVector
from qiskit import execute
from qiskit.circuit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.library import XGate, U3Gate
from typing import Union, Optional, Dict, List, Callable

class QASVM_circuit(QuantumCircuit):
    """ 
        Modulation of QASVM circuit (first order, second order) construction - Base Class:
        Example:
            qc = QASVM_circuit(num_data, dim_data, ord)
            qc.add_var_form(var_form)
            qc.UD_encode(feature_map, _feature_map_params, training_data=data, training_label=label, N=num_data)
            qc.X_encode(feature_map, _feature_map_params, testdata=testdata)
            qc.SWAP_test()
            qc.Z_expectation_measurement()
    """
    def __init__(self, num_data:int, dim_data:int, ord:int) -> None:
        assert (ord==1 or ord==2)
        super().__init__()
        regs = []
        regs.append(QuantumRegister(1, 'a'))
        regs.append(QuantumRegister(int(math.log2(num_data)), 'i'))
        regs.append(QuantumRegister(int(math.log2(dim_data)), 'xi'))
        regs.append(QuantumRegister(1, 'yi'))
        if ord==1:
            regs.append(QuantumRegister(int(math.log2(dim_data)), 'xj'))
            regs.append(ClassicalRegister(2, 'c'))
        else:
            regs.append(QuantumRegister(int(math.log2(num_data)), 'j'))
            regs.append(QuantumRegister(int(math.log2(dim_data)), 'xj'))
            regs.append(QuantumRegister(1, 'yj'))
            regs.append(ClassicalRegister(3, 'c'))
        super().__init__(*regs)
        self._reg_dict = {r.name:r for r in regs}
    
    def SWAP_test(self):
        self.h(self._reg_dict['a'])
        [self.cswap(self._reg_dict['a'], self._reg_dict['xi'][d], self._reg_dict['xj'][d]) for d in range(len(self._reg_dict['xi']))]
        self.h(self._reg_dict['a'])

    def Z_expectation_measurement(self):
        self.measure(list(self._reg_dict['a'])+list(self._reg_dict['yi']) + list(self._reg_dict.get('yj', [])), list(self._reg_dict['c']))

    def UD_encode(self, feature_map, feature_map_params, training_data:np.ndarray, training_label:np.ndarray, N:int, reg:str='i'):
        ctrl_q = self._reg_dict[reg]
        target_q = self._reg_dict['x'+reg]
        target_y = self._reg_dict['y'+reg]
        for ctrl_state in range(N):
            feature_map_param_dict = {p:v for p, v in zip(feature_map_params, training_data[ctrl_state])}
            bind_feature_map = feature_map.assign_parameters(feature_map_param_dict)
            self.append(bind_feature_map.to_gate().control(len(ctrl_q), ctrl_state=ctrl_state), qargs=list(ctrl_q)+list(target_q))
            if training_label[ctrl_state]<0.5:
                self.append(XGate().control(len(ctrl_q), ctrl_state=ctrl_state), qargs=list(ctrl_q)+list(target_y))

    def X_encode(self, feature_map, feature_map_params, testdata:Union[np.ndarray, ParameterVector, List[Parameter]], reg:str='xj'):
        feature_map_param_dict = {p:v for p, v in zip(feature_map_params, testdata)}
        self.compose(feature_map.assign_parameters(feature_map_param_dict), qubits=list(self._reg_dict[reg]), inplace=True)

    def add_var_form(self, var_form, reg:str='i'):
        self.compose(var_form, qubits=list(self._reg_dict[reg]), inplace=True)

class uniform_QASVM_circuit(QASVM_circuit):
    def add_var_form(self, var_form, reg:str='i'):
        self.h(self._reg_dict[reg])

class Bloch_sphere_QASVM_circuit(QASVM_circuit):
    def UD_encode(self, feature_map, feature_map_params, training_data:np.ndarray, training_label:np.ndarray, N:int, reg:str='i'):
        ctrl_q = self._reg_dict[reg]
        target_q = self._reg_dict['x'+reg]
        target_y = self._reg_dict['y'+reg]
        #self.uc([U3Gate(theta, phi, 0).to_matrix() for theta, phi in training_data], ctrl_q, target_q)
        self.ucry([theta for theta in training_data[:,0]], ctrl_q, target_q)
        self.ucrz([phi for phi in training_data[:,1]], ctrl_q, target_q)
        self.ucrx(list(np.where(training_label>0.5, 0, np.pi)), ctrl_q, target_y)

class Bloch_uniform_QASVM_circuit(Bloch_sphere_QASVM_circuit, uniform_QASVM_circuit):
    def add_var_form(self, var_form, reg: str):
        self.h(self._reg_dict[reg])

class _uc_QASVM_circuit(QASVM_circuit):
    def UD_encode(self, feature_map, feature_map_params, training_data:np.ndarray, training_label:np.ndarray, N:int, reg:str='i'):
        ctrl_q = self._reg_dict[reg]
        target_q = self._reg_dict['x'+reg]
        target_y = self._reg_dict['y'+reg]
        self.uc([U3Gate(theta, phi, 0).to_matrix() for theta, phi in training_data], ctrl_q, target_q)
        self.ucrx(list(np.where(training_label>0.5, 0, np.pi)), ctrl_q, target_y)