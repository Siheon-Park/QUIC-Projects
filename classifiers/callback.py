from abc import ABC, abstractmethod
from typing import Callable

from pandas import DataFrame, melt
from matplotlib import pyplot as plt
from seaborn import relplot, lineplot, scatterplot

from torch.utils.tensorboard import SummaryWriter

from . import Classifier
from .quantum.qasvm import ParameterArray


class CallBack(ABC):
    @abstractmethod
    def __call__(self, *args, **kwargs):
        raise NotImplementedError


CallBack.save = Classifier.save


class CostParamStorage(CallBack):
    """ saves simply costs and params"""

    def __init__(self) -> None:
        super().__init__()
        self.data = DataFrame()
        self.step = 0

    def __call__(self, nfev, parameters, cost, step_size, isaccepted):
        """Args: k, parameters, cost, step_size, isaccepted"""
        if isinstance(cost, Callable):
            cost = cost(parameters)

        if isinstance(parameters, ParameterArray):
            _temp_dict = dict(zip(map(str, parameters.parameter_vector), parameters))
        else:
            _temp_dict = dict(zip(map(str, range(len(parameters))), parameters))
        _temp_dict2 = dict(Step=self.step)
        _temp_dict2['# Func. Eval.'] = nfev
        _temp_dict2.update(_temp_dict)
        _temp_dict2['Cost'] = cost
        _temp_dict2['Step Size'] = step_size
        _temp_dict2['Accepted'] = isaccepted
        self.data = self.data.append(DataFrame(_temp_dict2, index=(self.step,)), ignore_index=False).sort_index()
        self.step += 1

    def add_writer(self, writer: SummaryWriter):
        k = self.data['Step'].iloc[-1]
        _temp_dict = self.data[self.data.columns[2:-3]].iloc[-1].to_dict()
        cost = self.data['Cost'].iloc[-1]
        isaccepted = self.data['Accepted'].iloc[-1]
        if cost is not None:
            writer.add_scalar('Cost', cost, k)
            writer.add_scalars('Parameters', _temp_dict, k)
            if isaccepted:
                writer.add_scalar('Cost(accepted)', cost, k)
                writer.add_scalars('Parameters(accepted)', _temp_dict, k)

    def plot_params(self, title: str = 'Parameters', ylabel: str = 'Value', method: str = 'relplot', ax=None, **kwargs):
        df = melt(self.data, id_vars=['Step', 'Accepted'], value_vars=self.data.columns[2:-3], var_name=title,
                  value_name=ylabel)
        if method == 'relplot':
            g = relplot(data=df, x='Step', y=ylabel, hue=title, style="Accepted", style_order=[True, False], **kwargs)
        elif method == 'mpl':
            if ax is None:
                ax = plt.gca()
            lineplot(data=df[df['Accepted'] == True], x='Step', y=ylabel, hue=title, style_order=[True, False], ax=ax,
                     legend=True, **kwargs)
            scatterplot(data=df[df['Accepted'] == False], x='Step', y=ylabel, hue=title, style="Accepted",
                        style_order=[True, False], ax=ax, legend=False, **kwargs)
            g = ax
        else:
            raise ValueError(f'No such method as {method}')
        plt.grid()

        return g

    def plot(self, method: str = 'relplot', ax=None, **kwargs):
        if method == 'relplot':
            g = relplot(data=self.data, x='Step', y='Cost', style='Accepted', style_order=[True, False], **kwargs)
        elif method == 'mpl':
            if ax is None:
                ax = plt.gca()
            lineplot(data=self.data[self.data['Accepted'] == True], x='Step', y='Cost', style='Accepted', ax=ax,
                     legend=True, **kwargs)
            scatterplot(data=self.data[self.data['Accepted'] == False], x='Step', y='Cost', style='Accepted', ax=ax,
                        legend=True, **kwargs)
            g = ax
        else:
            raise ValueError(f'No such method as {method}')
        return g

    def last_avg(self, last: int, ignore_rejected: bool = False):
        if not ignore_rejected:
            df = self.data
        else:
            df = self.data[self.data['Accepted'] == True]
        return df[df.columns[2:-3]][-last:].mean(axis=0).to_numpy()

    def last_std(self, last: int, ignore_rejected: bool = False):
        if not ignore_rejected:
            df = self.data
        else:
            df = self.data[self.data['Accepted'] == True]
        return df[df.columns[2:-3]][-last:].std(axis=0).to_numpy()

    def last_cost_avg(self, last: int, ignore_rejected: bool = False):
        if not ignore_rejected:
            df = self.data
        else:
            df = self.data[self.data['Accepted'] == True]
        return df['Cost'][-last:].mean()

    def last_cost_std(self, last: int, ignore_rejected: bool = False):
        if not ignore_rejected:
            df = self.data
        else:
            df = self.data[self.data['Accepted'] == True]
        return df['Cost'][-last:].std()

    def num_accepted(self):
        if len(self.data) == 0:
            return 0
        else:
            df = self.data[self.data['Accepted'] == True]
            return len(df)

    def termination_checker(self, minimum:float=None, last_avg:int=1, tol:float=0.1):
        """
            minimum: if float, terminate when obj. val. is less then minimum*(1+tol)
                     if None, terminate when obj. val. converges.
            last_avg: number of samples to average loss function
            tol: tol>=0
        """
        if minimum is None:
            def terminate(*args):
                if self.num_accepted() > 2 * last_avg and self.last_cost_avg(2 * last_avg, ignore_rejected=True) < self.last_cost_avg(last_avg, ignore_rejected=True):
                    return True
                else:
                    return False
        else:
            def terminate(*args):
                if self.num_accepted() > 2 * last_avg and self.last_cost_avg(last_avg, ignore_rejected=True) >= minimum*(1+tol):
                    return True
                else:
                    return False
        return terminate

    def __len__(self) -> int:
        return len(self.data)

if __name__=="__main__":
    storage = CostParamStorage()
    storage(0, [0, 0], 0, 0, True)
    print(len(storage))