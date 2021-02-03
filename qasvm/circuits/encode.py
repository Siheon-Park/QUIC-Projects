import numpy as np
from typing import List, Optional, Any, Iterable, Union
from qiskit.circuit import QuantumCircuit
from qiskit.circuit import QuantumRegister
from qiskit.circuit import Qubit
from qiskit.circuit import Instruction, Gate
from qiskit.circuit.library.standard_gates import XGate, ZGate
from qiskit.extensions.quantum_initializer.initializer import Initialize
from qiskit import transpile, QiskitError

class Encoder(Initialize):
    """Complex amplitude encoder.

    Simular to Initialize Instruction, but witout reset.
    definition is composed of cx and u3 gates, so it is unitary
    """

    def __init__(self, params, name="Encoder"):
        nparams = quantum_state(params).state
        super().__init__(nparams)
        #super().__init__(params)
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

        self.definition = encode_circuit
        #self.definition = transpile(encode_circuit, basis_gates=['cx', 'u3'])
        #self.definition = initialize_circuit

    def to_gate(self, parameter_map=None, label=None):
        gate = self.definition.to_gate(
            parameter_map=parameter_map, label=label)
        gate.name = self.name
        gate.params = np.real(self.params)
        return gate