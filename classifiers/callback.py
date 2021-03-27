from typing import Optional, List, Callable
import logging

import numpy as np
from matplotlib import pyplot as plt

from qiskit.aqua import aqua_globals
from qiskit.aqua.components.optimizers.spsa import SPSA

from torch.utils.tensorboard import SummaryWriter

from . import Classifier

from tqdm.notebook import tqdm
# TODO: how to properly save params
class CallBack(object):
    def __init__(self, writer:SummaryWriter=None) -> None:
        super().__init__()
        self.writer = writer

CallBack.save = Classifier.save

class SimpleStorage(CallBack):
    """ saves simply costs and params"""
    def __init__(self, writer:SummaryWriter=None) -> None:
        super().__init__(writer)
        self.params = {}
        self.costs = {}

    def __call__(self, *args, **kwargs):
        k, cost, theta = args[:3]
        if isinstance(cost, Callable):
            cost = cost(theta)
        self.params[k] = theta
        self.costs[k] = cost
        if self.writer is not None:
            self.writer.add_scalar('Cost', cost, k)
            self.writer.add_scalars('Parameters', dict(zip(map(str, range(len(theta))), theta)), k)

    def plot_params(self):
        params = np.array(list(self.params.values()))
        steps = np.array(list(self.params.keys()))
        [plt.plot(steps, params[:,i], label=f'parameter {i}') for i in range(params.shape[1])]
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.xlim([min(steps), max(steps)])
        plt.yticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi], ['-$\pi$', '-$\pi/2$', '$0$', '$\pi/2$', '$\pi$'])
        plt.xlabel('steps')
        plt.ylabel('param value')
        plt.grid()

    def plot(self):
        cost = np.array(list(self.costs.values()))
        steps = np.array(list(self.params.keys()))
        plt.plot(steps[0:], cost[0:], label='obj val', c='k', linewidth=0.5)
        plt.grid()
        plt.legend()

class SimplePMStorage(SimpleStorage):
    """ saves simply costs and params"""
    def __init__(self, writer:SummaryWriter=None) -> None:
        super().__init__(writer)
        self.params_pm = {}
        self.costs_pm = {}

    def __call__(self, *args, **kwargs):
        k, cost, theta, cost_plus, cost_minus, theta_plus, theta_minus = args[:7]
        super().__call__(k, cost, theta)
        if isinstance(cost_plus, Callable):
            cost_plus = cost_plus(theta_plus)
        if isinstance(cost_minus, Callable):
            cost_minus = cost_minus(theta_minus)
        self.params_pm[k] = [theta_minus, theta_plus]
        self.costs_pm[k] = [cost_minus, cost_plus]
        if self.writer is not None:
            self.writer.add_scalar('Cost/+', cost_plus, k)
            self.writer.add_scalar('Cost/-', cost_minus, k)

    def plot(self):
        cost_pm = np.array(list(self.costs_pm.values()))
        cost = np.array(list(self.costs.values()))
        steps = np.array(list(self.params.keys()))
        plt.scatter(steps[0:], cost_pm[0:][:,0], label='t-', c='b', s=1)
        plt.scatter(steps[0:], cost_pm[0:][:,1], label='t+', c='r', s=1)
        plt.plot(steps[0:], cost[0:], label='t0', c='k', linewidth=0.5)
        plt.grid()
        plt.legend()