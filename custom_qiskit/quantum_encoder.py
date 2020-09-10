""" Encoding gate creation """
import math
import numpy as np
from typing import List, Optional, Any
from qiskit.circuit import QuantumCircuit
from qiskit.circuit import QuantumRegister
from qiskit.circuit import Instruction, Gate
from qiskit.extensions.quantum_initializer.initializer import Initialize
from qiskit import transpile

from prepare_state import quantum_state

class Encoder(Initialize):
    """Complex amplitude encoder.

    Simular to Initialize Instruction, but witout reset.
    definition is composed of cx and u3 gates, so it is unitary
    """

    def __init__(self, params, name="Encoder"):
        nparams = quantum_state(params).state
        super().__init__(nparams)
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
        initialize_instr = disentangling_circuit.to_instruction().inverse()

        q = QuantumRegister(self.num_qubits, 'q')
        initialize_circuit = QuantumCircuit(q, name='init_def')
        #for qubit in q:
        #    initialize_circuit.append(Reset(), [qubit])
        initialize_circuit.append(initialize_instr, q[:])

        self.definition = transpile(initialize_circuit, basis_gates=['cx', 'u3'])
        #self.definition = initialize_circuit
    
    def to_gate(self, parameter_map=None, label=None):
        gate = self.definition.to_gate(parameter_map=parameter_map, label=label)
        gate.name = self.name
        return gate
# class EncoderGate(Gate):
#     """Coplex amplitude encoding gate"""
#     def __init__(self, state, name="Encoder"):
#         """Create new encoder"""
#         state, n = quantum_state(state)
#         super().__init__(name, n, state)

#     def _define(self):
#         self.definition = Encode(self.params, name=self.name).definition
