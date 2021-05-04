from abc import ABC, abstractmethod
from typing import Callable, Union

import numpy as np
from pandas import DataFrame, melt
from matplotlib import pyplot as plt
from seaborn import relplot, lineplot, scatterplot

from torch.utils.tensorboard import SummaryWriter
import logging

from . import Classifier

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
    def __init__(self, interval:int=1) -> None:
        super().__init__()
        self.intv = interval

    def __call__(self, k, parameters, cost, step_size, isaccepted):
        """Args: k, parameters, cost, step_size, isaccepted"""
        if k%self.intv != 0:
            cost=None
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

    def add_writer(self, writer:SummaryWriter):
        k = self.data['Step'].iloc[-1]
        _temp_dict = self.data[self.data.columns[1:-3]].iloc[-1].to_dict()
        cost = self.data['Cost'].iloc[-1]
        isaccepted = self.data['Accepted'].iloc[-1]
        if cost is not None:
            writer.add_scalar('Cost', cost, k)
            writer.add_scalars('Parameters', _temp_dict, k)
            if isaccepted:
                writer.add_scalar('Cost(accepted)', cost, k)
                writer.add_scalars('Parameters(accepted)', _temp_dict, k)

    def plot_params(self, title:str='Parameters', ylabel:str='Value', method:str='relplot', ax=None, **kwargs):
        df = melt(self.data, id_vars=['Step', 'Accepted'], value_vars=self.data.columns[1:-3], var_name=title, value_name=ylabel)
        if method=='relplot':
            g = relplot(data=df, x='Step', y=ylabel, hue=title, style="Accepted", style_order=[True, False], **kwargs)
        elif method=='mpl':
            if ax is None:
                ax = plt.gca()
            lineplot(data=df[df['Accepted']==True], x='Step', y=ylabel, hue=title, style_order=[True, False], ax=ax, legend=True, **kwargs)
            scatterplot(data=df[df['Accepted']==False], x='Step', y=ylabel, hue=title, style="Accepted", style_order=[True, False], ax=ax, legend=False, **kwargs)
            g=ax
        else:
            raise ValueError(f'No such method as {method}')
        plt.grid()

        return g

    def plot(self, method:str='relplot', ax=None, **kwargs):
        if method=='relplot':
            g=relplot(data=self.data, x='Step', y='Cost', style='Accepted', style_order=[True, False], **kwargs)
        elif method=='mpl':
            if ax is None:
                ax = plt.gca()
            lineplot(data=self.data[self.data['Accepted']==True], x='Step', y='Cost', style='Accepted', ax=ax, legend=True, **kwargs)
            scatterplot(data=self.data[self.data['Accepted']==False], x='Step', y='Cost', style='Accepted', ax=ax, legend=True, **kwargs)
            g=ax
        else:
            raise ValueError(f'No such method as {method}')
        return g

    def last_avg(self, last:int):
        return self.data[self.data.columns[1:-3]][-last:].mean(axis=0).to_numpy()

    def last_std(self, last:int):
        return self.data[self.data.columns[1:-3]][-last:].std(axis=0).to_numpy()

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

class BaseJobCallback(CallBack):
    pass

class JobLogger(BaseJobCallback):
    def __init__(self, logfile:str='./job_log.log', loglevel:int=logging.INFO) -> None:
        super().__init__()
        self.logger = logging.getLogger('job_logger')
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler = logging.FileHandler(logfile)
        handler.setLevel(loglevel)
        handler.setFormatter(formatter)
        self.logger.setLevel(loglevel)
        self.logger.addHandler(handler)

    def __call__(self, job_id, job_status, queue_position, job):
        pass
