from qiskit.aqua.algorithms.vq_algorithm import VQResult
from qiskit.aqua.components.optimizers.optimizer import Optimizer
from qiskit.aqua.components.optimizers import SPSA
from classifiers.quantum import QuantumError
import logging
import numpy as np
from numpy.core.fromnumeric import var
from qiskit.aqua.quantum_instance import QuantumInstance
from qiskit.aqua import aqua_globals
from qiskit.circuit.quantumcircuit import QuantumCircuit
import torch
from torch.autograd import Function
from torch import optim
from torch import nn
from torch.nn import functional as F
from .qasvm import QASVM
from typing import Any, Callable, Optional, Union

logger = logging.getLogger(__name__)

class GeneralFunction(Function):

    @staticmethod
    def forward(ctx, params, cost_fn:Callable, grad_fn:Callable):
        ctx.cost_fn = cost_fn
        ctx.grad_fn = grad_fn
        evaluation = torch.tensor([ctx.cost_fn(params.tolist())])
        ctx.save_for_backward(params)
        return evaluation

    @staticmethod
    def backward(ctx, grad_output):
        params, = ctx.saved_tensors

        if ctx.grad_fn is not None:
            gradients = ctx.grad_fn(params.tolist())
        else:
            input_list = np.array(params.tolist())
            input_list_rights = input_list + 1e-3*np.eye(len(input_list))
            input_list_lefts = input_list + 1e-3*np.eye(len(input_list))

            gradients = (np.array(list(map(ctx.cost_fn, input_list_rights))) - np.array(list(map(ctx.cost_fn, input_list_lefts))))/(2*1e-3)

        return grad_output * gradients, None, None

class QASVM_model(nn.Module):
    def __init__(self, qasvm:QASVM) -> None:
        super().__init__()
        self.qasvm = qasvm
        if self.qasvm.initial_point is not None:
            self.params = nn.Parameter(self.qasvm.initial_point)
        else:
            self.params = nn.Parameter(torch.randn(self.qasvm.num_parameters))

    def forward(self):
        return GeneralFunction.apply(self.params, self.qasvm.cost_fn, self.qasvm.grad_fn)

    def update_qasvm_result(self):
        with torch.no_grad():
            param_list = self.params.data.numpy()
            result = VQResult()
            result.optimizer_evals = None
            result.optimizer_time = None
            result.optimal_value = self.qasvm.cost_fn(param_list)
            result.optimal_point = param_list
            result.optimal_parameters = dict(zip(self.qasvm.var_form_params['0'], param_list))
            self.qasvm.result = result


class StocaticOptimizer(optim.Optimizer):
    pass
# FIXME: fucking asshole
class Spsa(StocaticOptimizer):

    _C0 = 2 * np.pi * 0.1
    _OPTIONS = ['save_steps', 'last_avg']

    def __init__(self, params, cost_fn:Callable, 
                 maxiter: int = 1000,
                 c0: float = _C0,
                 c1: float = 0.1,
                 c2: float = 0.602,
                 c3: float = 0.101,
                 c4: float = 0,
                 skip_calibration: bool = False) -> None:
        spsa = SPSA(maxiter=maxiter, c0=c0, c1=c1, c2=c2, c3=c3, c4=c4, skip_calibration=skip_calibration)
        self._skip_cal = skip_calibration
        default = dict(c0=c0, c1=c1, c2=c2, c3=c3, c4=c4, cost_fn=cost_fn, maxiter=maxiter, step=0)
        super().__init__(params, default)
        if not spsa._skip_calibration:
            with torch.no_grad():
                num_steps_calibration = min(25, max(1, spsa._maxiter // 5))
                for group in self.param_groups:
                    spsa._calibration(cost_fn, group['params'][0].data.numpy(), num_steps_calibration)
                    group['c0'], group['c1'], group['c2'], group['c3'], group['c4'] = spsa._parameters

    @torch.no_grad()
    def step(self, closure:Callable=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            theta = group['params'][0]
            # SPSA Parameters
            a_spsa = float(group['c0']) / np.power(group['step'] + 1 + group['c4'],
                                                           group['c2'])
            c_spsa = float(group['c1']) / np.power(group['step'] + 1, group['c3'])
            delta = 2 * aqua_globals.random.integers(2, size=theta.shape[0]) - 1
            # plus and minus directions
            theta_plus = theta + c_spsa * delta
            theta_minus = theta - c_spsa * delta
            # cost function for the two directions

            cost_plus = group['cost_fn'](theta_plus.data.numpy())
            cost_minus = group['cost_fn'](theta_minus.data.numpy())
            # derivative estimate
            g_spsa = (cost_plus - cost_minus) * delta / (2.0 * c_spsa)
            # updated theta
            theta = theta - a_spsa * g_spsa
            group['step']+=1
            self.state[theta]

        return loss

