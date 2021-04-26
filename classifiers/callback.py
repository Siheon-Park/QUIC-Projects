from abc import ABC, abstractmethod
from typing import Callable, Dict, Union

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
    def __init__(self) -> None:
        super().__init__()
        self.data = DataFrame()

    def clear(self):
        del self.data
        self.data = DataFrame()

class CostParamStorage(BaseStorage):
    """ saves simply costs and params"""
    def __init__(self) -> None:
        super().__init__()

    def __call__(self, k, parameters, cost, step_size, isaccepted, **kwargs):
        """Args: k, parameters, cost, step_size, isaccepted"""
        writer = kwargs.get('writer', None)

        if isinstance(cost, Callable):
            cost = cost(parameters)

        if isinstance(parameters, dict):
            _temp_dict = dict(zip(map(str, parameters.keys()), parameters.values()))
        else:
            _temp_dict = dict(zip(map(str, range(len(parameters))), parameters))
        _temp_dict2 = dict(Step=k)
        _temp_dict2.update(_temp_dict)
        _temp_dict2['Cost'] = cost
        _temp_dict2['Step Size'] = step_size
        _temp_dict2['Accepted'] = isaccepted
        self.data = self.data.append(DataFrame(_temp_dict2, index=[k]), ignore_index=False).sort_index()

        if writer is not None:
            writer.add_scalar('Cost', cost, k)
            writer.add_scalars('Parameters', _temp_dict, k)
            if isaccepted:
                writer.add_scalar('Cost(accepted)', cost, k)
                writer.add_scalars('Parameters(accepted)', _temp_dict, k)

    def plot_params(self, ax=None, title='Parameters', linewidth=1, axis_labels=('steps', None)):
        if ax is None:
            ax = plt.gca()
        isaccepted = self.data['Accepted']=='True'
        df = self.data.drop(['Cost', 'Step Size', 'Accepted'], axis=1).loc[isaccepted]
        ret = df.plot(kind='line', ax=ax, grid=True, title=title, legend=False, linewidth=linewidth)
        ax.legend(bbox_to_anchor=(1.02, 1.02))
        ax.set_xlabel(axis_labels[0])
        ax.set_ylabel(axis_labels[1])
        return ret

    def plot(self, ax=None, title='Costs', linewidth=0.5, s=1, c='k', label='Cost', axis_labels=('steps', None)):
        if ax is None:
            ax = plt.gca()
        isaccepted = self.data['Accepted']=='True'
        ret = self.data.loc[isaccepted].plot(y='Cost', kind='line', ax=ax, grid=True, title=title, legend=False, label=label, linewidth=linewidth, c=c[0])
        ax.legend()
        ax.set_xlabel(axis_labels[0])
        ax.set_ylabel(axis_labels[1])
        return ret

class BaseStopping(CallBack):
    pass

class ParamsStopping(BaseStopping):
    def __init__(self, last_avg:int=30, patiance:int=None, tol:float=1e-2) -> None:
        if patiance is None:
            patiance = last_avg
        assert patiance>=0
        assert last_avg>=2
        assert tol>0
        super().__init__()
        self.watch_list = DataFrame()
        self.patiance = patiance
        self.last_avg = last_avg
        self.tol = tol
        self._FLAG = -1
        self.best_params = None

    def __call__(self, params:Union[np.ndarray, dict], step:int):
        if isinstance(params, dict):
            _temp_dict = dict(zip(map(str, params.keys()), params.values()))
        else:
            _temp_dict = dict(zip(map(str, range(len(params))), params))
        if step%self.last_avg not in self.watch_list.index:
            self.watch_list = self.watch_list.append(DataFrame(_temp_dict, index=[step%self.last_avg]), ignore_index=False)
        else:
            self.watch_list.loc[[step%self.last_avg]] = DataFrame(_temp_dict, index=[step%self.last_avg])
        self.best_params = self.watch_list.mean(axis=0).to_numpy()
        if len(self.watch_list)>=self.last_avg:
            if (self.watch_list.std(axis=0)<self.tol).all():
                self._FLAG+=1
                if self._FLAG==self.patiance:
                    return True
        return False

class LastAvgStorage(CallBack):
    def __init__(self, last_avg:int=30) -> None:
        super().__init__()
        self.watch_list = DataFrame()
        self.last_avg = last_avg

    def __call__(self, params:Union[np.ndarray, dict], step:int):
        if isinstance(params, dict):
            _temp_dict = dict(zip(map(str, params.keys()), params.values()))
        else:
            _temp_dict = dict(zip(map(str, range(len(params))), params))
        if step%self.last_avg not in self.watch_list.index:
            self.watch_list = self.watch_list.append(DataFrame(_temp_dict, index=[step%self.last_avg]), ignore_index=False)
        else:
            self.watch_list.loc[[step%self.last_avg]] = DataFrame(_temp_dict, index=[step%self.last_avg])

    @property
    def best_params(self):
        return self.watch_list.mean(axis=0).to_numpy()