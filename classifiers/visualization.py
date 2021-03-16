from matplotlib import pyplot as plt
import numpy as np
import re
import pathlib
from itertools import product
from typing import Union, Optional, Dict, List, Callable
from .convex.svm import Classifier
from .quantum.qasvm import QASVM
from .utils import tsne

class Plot_Log_From_SPSA(object):
    def __init__(self, logfile:pathlib.Path=None) -> None:
        super().__init__()
        if logfile is None:
            q = pathlib.Path.home()/ 'QUIC-Projects' / 'exp_logs'
            logfile = max(q.glob('*.log'), key=lambda q: q.stat().st_ctime)
        self.logfile = logfile
        self.theta_plus_re = re.compile('Objective function at theta[+] for step # \d+: -?\d+[.]\d+')
        self.theta_minus_re = re.compile('Objective function at theta[-] for step # \d+: -?\d+[.]\d+')
        self.theta_plus_dict = dict()
        self.theta_minus_dict = dict()
        with open(logfile, 'r') as f:
            lines = f.readlines()
            for line in lines:
                theta_minus_search = self.theta_minus_re.search(line)
                theta_plus_search = self.theta_plus_re.search(line)
                if theta_minus_search is not None:
                    theta_minus_str = theta_minus_search.group()
                    theta_minus_list = re.findall(r'\d+', theta_minus_str)
                    theta_minus = float('{:}.{:}'.format(theta_minus_list[1], theta_minus_list[2]))
                    index = int(theta_minus_list[0])
                    self.theta_minus_dict[index] = theta_minus
                if theta_plus_search is not None:
                    theta_plus_str = theta_plus_search.group()
                    theta_plus_list = re.findall(r'\d+', theta_plus_str)
                    theta_plus = float('{:}.{:}'.format(theta_plus_list[1], theta_plus_list[2]))
                    index = int(theta_plus_list[0])
                    self.theta_plus_dict[index] = theta_plus

    def plot(self, **kwargs):
        psteps = np.array(list(self.theta_plus_dict.keys()))
        msteps = np.array(list(self.theta_minus_dict.keys()))
        pval = np.array(list(self.theta_plus_dict.values()))
        mval = np.array(list(self.theta_minus_dict.values()))
        min_step = kwargs.get('min_step', 0)
        max_step = kwargs.get('max_step', max(psteps))
        pind = (psteps>=min_step)*(psteps<=max_step)
        mind = (msteps>=min_step)*(msteps<=max_step)
        plt.scatter(psteps[pind], pval[pind], label='θ+', c='b', s=1)
        plt.scatter(msteps[mind], mval[mind], label='θ-', c='r', s=1)
        plt.legend()
        plt.grid()
        plt.title(kwargs.get('title', 'SPSA optimization'))

class Plot_SVM(object):
    def __init__(self, cls:Union[Classifier, QASVM]) -> None:
        super().__init__()
        self.cls = cls

    def plot(self, option:str='sv', ax=plt, a = (0,1), *args, **kwargs):
        if 'tsne' in option:
            data = tsne(self.cls.data, kwargs.get('perp', 30), kwargs.get('seed', 0))
            a = (0, 1)
        else:
            data = self.cls.data

        if 'data' in option:
            ax.scatter(data[:,a[0]], data[:,a[1]], c=self.cls.label, *args, **kwargs)
        elif 'sv' in option:
            if 'cmap' not in kwargs:
                kwargs['cmap'] = plt.cm.coolwarm
            support_vector = data[self.cls.support_]
            ax.scatter(data[:,a[0]], data[:,a[1]], c=self.cls.label, cmap = plt.cm.coolwarm)
            ax.scatter(support_vector[:,a[0]], support_vector[:,a[1]], s=100, linewidth=1.0, edgecolors='k', facecolors='none')
        elif 'density' in option:
            sc = ax.scatter(data[:,a[0]], data[:,a[1]], c=self.cls.alpha*self.cls.label, *args, **kwargs)
            if ax==plt:
                plt.colorbar()
            else:
                plt.colorbar(sc, ax=ax)
        elif 'alpha' in option:
            ax.plot(self.cls.alpha, 'k', *args, **kwargs)
            ax.plot(self.cls.support_, self.cls.alpha[self.cls.support_], 'xk', *args, **kwargs)
        else:
            raise ValueError('invalid option')

        if ax==plt:
            ax.title(f'{self.cls.name}')
        else:
            ax.set_title(f'{self.cls.name}')
        ax.grid()

    def plot_boundary(self, ax=plt, plot_data:bool=True, fig=None, color_setting:dict={}):
        assert self.cls.data.shape[1]==2
        xx = np.linspace(min(self.cls.data[:,0]), max(self.cls.data[:,0]), 100)
        yy = np.linspace(min(self.cls.data[:,1]), max(self.cls.data[:,1]), 10)
        XX, YY = np.meshgrid(xx, yy)
        xxx = XX.flatten()
        yyy = YY.flatten()
        zzz = self.cls.f(np.vstack((xxx,yyy)).T)
        ZZ = zzz.reshape(XX.shape)
        c1 = min(np.abs(self.cls.f(self.cls.data[self.cls.polary>0])))
        c2 = -min(np.abs(self.cls.f(self.cls.data[self.cls.polary<0])))
        if not color_setting:
            cdict = {-1:'b', 0:'k', 1:'r', c1:'m', c2:'c'}
        else:
            cdict = {}
            for k,v in color_setting.items():
                if k=='c1':
                    cdict[c1] = v
                elif k=='c2':
                    cdict[c2] = v
                else:
                    cdict[k] = v
        levels = dict(sorted(cdict.items()))
        CS = ax.contour(XX, YY, ZZ, levels=tuple(levels.keys()), colors=tuple(levels.values()))
        PS = ax.contourf(XX, YY, ZZ, levels = 100, cmap='RdBu')
        ax.clabel(CS, inline=1, fontsize=20)
        if fig is None:
            ax.colorbar(PS)
        else:
            fig.colorbar(PS, ax=ax)
        if plot_data:
            self.plot('sv', ax=ax)

class Plot_Data(object):
    def __init__(self, X:np.ndarray, y:np.ndarray) -> None:
        super().__init__()
        self.data = X
        self.label = y

    def plot(self, ax=plt, a=(0, 1), *args, **kwargs):
        ax.scatter(self.data[:,a[0]], self.data[:,a[1]], c=self.label, cmap='RdBu', *args, **kwargs)
        if ax==plt:
            ax.title('Data Distribution')
        else:
            ax.set_title('Data Distribution')
        ax.grid()

    
