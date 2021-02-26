import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, KernelCenterer
from sklearn.metrics import pairwise_kernels

class DataScaler(object):
    def __init__(self, scaling:str='standard', *args, **kwargs):
        slist = ['standard', 'minmax', 'maxabs', 'robust', 'normalize', 'kernel']
        if scaling not in slist:
            raise ValueError('Expect {:}, received {:}'.format(slist, scaling))
        if scaling == 'standard':
            self.scaler = StandardScaler(*args, **kwargs)
        elif scaling == 'minmax':
            self.scaler = MinMaxScaler(*args, **kwargs)
        elif scaling == 'maxabs':
            #self.scaler = MaxAbsScaler(*args, **kwargs)
            self.scaler = MinMaxScaler(feature_range=(-1,1), *args, **kwargs)
        elif scaling == 'robust':
            self.scaler = RobustScaler(*args, **kwargs)
        elif scaling == 'normalize':
            self.scaler = NormalizeTransformer(*args, **kwargs)
        else:
            self.scaler = KernelStandardScaler(*args, **kwargs)

    def __call__(self, X):
        return self.fit_transform(X)
    
    def fit(self, X):
        self.scaler.fit(X)

    def transform(self, X):
        return self.scaler.transform(X)

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

    def inverse_transform(self, X):
        return self.scaler.inverse_transform(X)

class NormalizeTransformer(DataScaler):
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    def fit(self, X):
        self.lens = np.linalg.norm(X, *self.args, **self.kwargs, axis=1)

    def transform(self, X):
        return (X.T/self.lens).T

    def inverse_transform(self, X):
        return (X.T*self.lens).T

class KernelStandardScaler(DataScaler):
    def __init__(self, kernel):
        self.kernel = kernel
        self.scaler = KernelCenterer()

    def fit(self, X):
        K = pairwise_kernels(X, X, metric=self.kernel)
        super().fit(K)

    def transform(self, X):
        K = pairwise_kernels(X, X, metric=self.kernel)
        return super().transform(K)

    def inverse_transform(self, X):
        pass

class DataMultiScaler(DataScaler):
    def __init__(self, *args):
        self.scalers = args

    def __call__(self, X):
        for scaler in self.scalers:
            X = scaler(X)
        return X
