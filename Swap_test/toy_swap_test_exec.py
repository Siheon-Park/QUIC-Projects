#%%
import sys
import os
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit import Aer, IBMQ, execute
from qiskit.circuit import Parameter
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display
# for user define modules
module_path = os.path.join('/Users/shaun/')
if module_path not in sys.path:
    sys.path.append(module_path)

import Swap_gate_classifier.toy_swap_test as toy

#%%
theta = Parameter('theta')
alpha = Parameter('alpha')
q = QuantumRegister(5, 'q')
c = ClassicalRegister(2, 'c')
a, m, x, y, t = (q) # adjust to backend
qc = QuantumCircuit(q, c)
qc.append(toy.weighting_gate(alpha), [m])
qc.append(toy.training_gate(), [m, x, y])
qc.append(toy.test_encoding_gate(theta), [t])
qc.append(toy.swap_classifier_gate(), [a, x, t])
qc.measure([a, y], c)
display(qc.draw('mpl'))

#%%
thetas = np.linspace(0, 2*np.pi, 128)
alphas = np.linspace(0, 2*np.pi, 128)

backend = Aer.get_backend('qasm_simulator')
job = execute(qc, backend=backend, shots=2**10,
              parameter_binds=[{theta: tval, alpha: np.pi/2} for tval in thetas])
result = job.result().get_counts()
#%%
corr = toy.swap_test_postprocess(result)
plt.plot(thetas, corr)
plt.ylim([-1, 1])
plt.xlim([0, 2*np.pi])
plt.grid()
plt.show()
