from abc import ABC, abstractmethod
from typing import Callable, Union

import numpy as np
from pandas import DataFrame
from matplotlib import pyplot as plt

from torch.utils.tensorboard import SummaryWriter

from . import Classifier
from .quantum.qasvm import ParameterDict

class CallBack(ABC):
    @abstractmethod
    def __call__(self):
        raise NotImplementedError

CallBack.save = Classifier.save

class BaseStorage(CallBack):
    def __init__(self, writer: SummaryWriter) -> None:
        super().__init__()
        self.writer=writer
        self.data = DataFrame()

    def clear(self):
        del self.data
        self.data = DataFrame()

class CostParamStorage(BaseStorage):
    """ saves simply costs and params"""
    def __init__(self, writer:SummaryWriter=None) -> None:
        super().__init__(writer)

    def __call__(self, *args, **kwargs):
        if len(args)==3:
            k, cost, theta = args
            cost_plus, cost_minus, theta_plus, theta_minus = None, None, None, None
        elif len(args)==5:
            k, cost, theta, cost_plus, cost_minus = args
            theta_plus, theta_minus = None, None
        elif len(args)==7:
            k, cost, theta, cost_plus, cost_minus, theta_plus, theta_minus = args
        else:
            raise ValueError('Args: step, cost, theta[, cost_plus, cost_minus[, theta_plus, theta_minus]]')
        if isinstance(cost, Callable):
            cost = cost(theta)
        if isinstance(cost_minus, Callable):
            cost_minus = cost_minus(theta_minus)
        if isinstance(cost_plus, Callable):
            cost_plus = cost_plus(theta_plus)

        if isinstance(theta, dict):
            _temp_dict = dict(zip(map(str, theta.keys()), theta.values()))
        else:
            _temp_dict = dict(zip(map(str, range(len(theta))), theta))
        if cost is not None:
            _temp_dict['Cost'] = cost
        if cost_plus is not None:
            _temp_dict['+'] = cost_plus
        if cost_minus is not None:
            _temp_dict['-'] = cost_minus
        self.data = self.data.append(DataFrame(_temp_dict, index=[k]), ignore_index=False)
        self.data.sort_index()

        if self.writer is not None:
            self.writer.add_scalar('Cost', cost, k)
            if cost_plus is not None:
                self.writer.add_scalar('Cost/+', cost_plus, k)
            if cost_minus is not None:
                self.writer.add_scalar('Cost/-', cost_minus, k)
            self.writer.add_scalars('Parameters', _temp_dict, k)

    def plot_params(self, ax=None, title='Parameters', linewidth=1, axis_labels=('steps', None)):
        if ax is None:
            ax = plt.gca()
        try:
            df = self.data.drop(['Cost', '+', '-'], axis=1)
        except KeyError:
            df = self.data.drop(['Cost'], axis=1)
        df.plot(kind='line', ax=ax, grid=True, title=title, legend=False, linewidth=linewidth)
        ax.legend(bbox_to_anchor=(1.02, 1.02))
        ax.set_xlabel(axis_labels[0])
        ax.set_ylabel(axis_labels[1])

    def plot(self, ax=None, title='Costs', linewidth=0.5, s=1, c=('k', 'r', 'b'), label=('Cost', '+', '-'), axis_labels=('steps', None)):
        if ax is None:
            ax = plt.gca()
        self.data.plot(y='Cost', kind='line', ax=ax, grid=True, title=title, legend=False, label=label[0], linewidth=linewidth, c=c[0])
        if '+' in self.data.columns:
            ax.scatter(self.data.index.to_numpy(), self.data['+'].to_numpy(), label=label[1], c=c[1], s=s)
        if '-' in self.data.columns:
            ax.scatter(self.data.index.to_numpy(), self.data['-'].to_numpy(), label=label[2], c=c[2], s=1)
        ax.legend()
        ax.set_xlabel(axis_labels[0])
        ax.set_ylabel(axis_labels[1])

class BaseStopping(CallBack):
    pass

class ParamsStopping(BaseStopping):
    def __init__(self, patiance:int=30, last_avg:int=30, tol:float=1e-2) -> None:
        assert patiance>=0
        assert last_avg>=2
        assert tol>0
        super().__init__()
        self.watch_list = DataFrame()
        self.patiance = patiance
        self.last_avg = last_avg
        self.tol = tol
        self._FLAG = -1
        self.best_params = ParameterDict()

    def __call__(self, params:Union[np.ndarray, dict], step:int):
        if isinstance(params, dict):
            _temp_dict = dict(zip(map(str, params.keys()), params.values()))
        else:
            _temp_dict = dict(zip(map(str, range(len(params))), params))
        if step%self.last_avg not in self.watch_list.index:
            self.watch_list.append(DataFrame(_temp_dict, index=[step%self.last_avg]), ignore_index=False)
        else:
            self.watch_list.loc[[step%self.last_avg]] = DataFrame(_temp_dict, index=[step%self.last_avg])
        if len(self.watch_list)>=self.last_avg:
            if (self.watch_list.std(axis=0)<self.tol).all():
                self._FLAG+=1
                if self._FLAG==self.patiance:
                    self.best_params = self.watch_list.mean(axis=0).to_numpy()
        return self.best_params