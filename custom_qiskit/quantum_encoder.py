""" Encoding gate creation """
import math
import numpy as np
from typing import List, Optional, Any, Iterable, Union
from qiskit.circuit import QuantumCircuit
from qiskit.circuit import QuantumRegister
from qiskit.circuit import Qubit
from qiskit.circuit import Instruction, Gate
from qiskit.circuit.library.standard_gates import XGate, ZGate
from qiskit.extensions.quantum_initializer.initializer import Initialize
from qiskit import transpile, QiskitError

# custom qiskit
from .prepare_state import quantum_state


class Encoder(Initialize):
    """Complex amplitude encoder.

    Simular to Initialize Instruction, but witout reset.
    definition is composed of cx and u3 gates, so it is unitary
    """

    def __init__(self, params, name="Encoder"):
        nparams = quantum_state(params).state
        super().__init__(nparams)
        super().__init__(params)
        self.name = name
        self.original_data = params

    def _define(self):
        """Calculate a subcircuit that implements this initialization

        Implements a recursive initialization algorithm, including optimizations,
        from "Synthesis of Quantum Logic Circuits" Shende, Bullock, Markov
        https://arxiv.org/abs/quant-ph/0406176v5

        Additionally implements some extra optimizations: remove zero rotations and
        double cnots.
        """
        # call to generate the circuit that takes the desired vector to zero
        disentangling_circuit = self.gates_to_uncompute()

        # invert the circuit to create the desired vector from zero (assuming
        # the qubits are in the zero state)
        encode_instr = disentangling_circuit.to_instruction().inverse()

        q = QuantumRegister(self.num_qubits, 'q')
        encode_circuit = QuantumCircuit(q, name='encode_def')
        # for qubit in q:
        #    initialize_circuit.append(Reset(), [qubit])
        encode_circuit.append(encode_instr, q[:])

        self.definition = transpile(encode_circuit, basis_gates=['cx', 'u3'])
        #self.definition = initialize_circuit

    def to_gate(self, parameter_map=None, label=None):
        gate = self.definition.to_gate(
            parameter_map=parameter_map, label=label)
        gate.name = self.name
        gate.params = np.real(self.params)
        return gate

def _valid_qubits(self, qubits)->List:
    """qubits:int, QuantumRegister, Qubit"""
    _qubits = []
    for q in qubits:
        if isinstance(q, QuantumRegister):
            for qq in q:
                _qubits.append(qq)
        elif isinstance(q, int):
            _qubits.append(self.qubits[q])
        elif isinstance(q, Qubit):
            _qubits.append(q)
        else:
            raise QiskitError('invalid qubits repr')
    return _qubits

def encode(self, state: List, qubits:Any, name: str = "Encoder") -> None:
    """append encoder(state) to 'qubits:int, QuantumRegister, Qubit' (self: QuantumCircuit)"""
    # qubits should be iter(qubits)
    qubits = [qubits] if not isinstance(qubits, Iterable) else qubits # case if qubits are not iterable
    
    # case if qubits are iter(qubits)
    # case if qubits are iter(qr)
    _qubits = _valid_qubits(self, qubits)
    self.append(Encoder(state, name=name), _qubits)

QuantumCircuit.encode = encode


def ctrl_encode(self, state: List, ctrl_state:Union[int, str, None], base:Any, ctrl:Any, name:str="Encoder", ) -> None:
    """
        append encoder(state) to 'qubits' with control 'ctrl_qubits' when control state is 'ctrl_state'
    """
    _base = _valid_qubits(self, base)
    _ctrl = _valid_qubits(self, ctrl)
    _qubits = _ctrl+_base
    encoder_ctrl_gate = Encoder(state, name=name).to_gate().control(
        num_ctrl_qubits=np.size(_ctrl),
        ctrl_state=ctrl_state)
    self.append(encoder_ctrl_gate, _qubits)

QuantumCircuit.ctrl_encode = ctrl_encode

def ctrl_x(self, ctrl_state:Union[int, str, None], base:Any, ctrl:Any)->None:
    """
        append X to 'qubits' with control 'ctrl_qubits' when control state is 'ctrl_state'
    """
    _base = _valid_qubits(self, base)
    _ctrl = _valid_qubits(self, ctrl)
    _qubits = _ctrl+_base    
    _gate = XGate().control(
        num_ctrl_qubits=np.size(_ctrl),
        ctrl_state=ctrl_state)
    self.append(_gate, _qubits)

QuantumCircuit.ctrl_x = ctrl_x

def ctrl_z(self, ctrl_state:Union[int, str, None], base:Any, ctrl:Any)->None:
    """
        append Z to 'qubits' with control 'ctrl_qubits' when control state is 'ctrl_state'
    """
    _base = _valid_qubits(self, base)
    _ctrl = _valid_qubits(self, ctrl)
    _qubits = _ctrl+_base    
    _gate = ZGate().control(
        num_ctrl_qubits=np.size(_qubits),
        ctrl_state=ctrl_state)
    self.append(_gate, _qubits)

QuantumCircuit.ctrl_z = ctrl_z
