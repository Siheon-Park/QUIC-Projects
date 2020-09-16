import math
import numpy as np
from typing import List, Optional, Any, Dict
from qiskit.exceptions import QiskitError
from qiskit.circuit import QuantumCircuit
from qiskit.circuit import QuantumRegister
from qiskit.circuit import ClassicalRegister
from qiskit.circuit import Instruction, Gate
from qiskit.circuit.library.standard_gates.x import XGate
from qiskit.extensions.quantum_initializer.initializer import Initialize
from qiskit import transpile

from prepare_state import quantum_state
from quantum_encoder import Encoder

class QuantumQuadraticProgrammingCircuit(Instruction):
    def __init__(self):
        
