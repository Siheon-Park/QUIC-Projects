from sys import flags
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer, normalize, StandardScaler, MaxAbsScaler, MinMaxScaler, RobustScaler
from matplotlib import cm, pyplot as plt

class DataScaler(object):
    def __init__(self, scaling:str='standard', *args, **kwargs):
        slist = ['standard', 'minmax', 'maxabs', 'robust']
        if scaling not in slist:
            raise ValueError('Expect {:}, received {:}'.format(slist, scaling))
        if scaling == 'standard':
            self.scaler = StandardScaler(*args, **kwargs)
        elif scaling == 'minmax':
            self.scaler = MinMaxScaler(*args, **kwargs)
        elif scaling == 'maxabs':
            #self.scaler = MaxAbsScaler(*args, **kwargs)
            self.scaler = MinMaxScaler(feature_range=(-1,1), *args, **kwargs)
        else:
            self.scaler = RobustScaler(*args, **kwargs)

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

class DataMultiScaler(DataScaler):
    def __init__(self, *args):
        self.scalers = args

    def __call__(self, X):
        for scaler in self.scalers:
            X = scaler(X)
        return X

class DataLoader:
    pass

class Sklearn_DataLoader(DataLoader):
    """
        Dataset loader for iris, wine data provided in Scikit-learn package

        Attrbutes:
            lb - LabelBinarizer()
            yhot - one-hot encoded label of original data
            data - transformed data
            label - transformed binary label (0, 1)
    """
    def __init__(self, dataset:str='iris', labels:tuple=None) -> None:
        """
            Args:
                dataset:str='iris' - type of dataset. i.e. 'iris', 'wine'
                mean:np.ndarray=0 - set the avarge point of the data. default is origin
                normalize:int=2 - normalize each data. see argument ord of numpy.linalg.norm.
                labels:tuple=None - labels to be considered. if None(default), select all labels in data
        """
        super().__init__()
        if dataset=='iris':
            ds = datasets.load_iris
        elif dataset=='wine':
            ds = datasets.load_wine
        else:
            raise NotValidDataTypeError
        data, label = ds(return_X_y=True)

        if labels is not None:
            _temp = [t in labels for t in label]
            data = data[_temp]
            label = label[_temp]
        
        self.limit_num_data = len(label)

        self.lb = LabelBinarizer()
        self.yhot = self.lb.fit_transform(label)
        self.data = data

    def __call__(self, num_data:int, true_hot:int, shuffle:bool=True):
        """
            Args:
                num_data:int - number of training data
                true_hot:int - label to be encoded as 1
                shuffle:bool=True - to shuffle training, test dataset
            
            return:
                X, y, Xt, yt - training data, training label, test data, test label
        """
        assert num_data>0
        self.label = self.yhot[:,true_hot].flatten()
        if num_data < self.limit_num_data:
            X, Xt, y, yt = train_test_split(self.data, self.label, train_size = num_data, shuffle=shuffle, stratify=self.label)
        else:
            X = self.data
            y = self.label
            Xt = None
            yt = None
        return X, y, Xt, yt

class Toy2DLinearLoader(DataLoader):
    def __init__(self, w:tuple, b:float) -> None:
        assert len(w)==2
        self.w = np.array(w).reshape(-1,1)
        self.b = b

        self.A = np.array(
            [
                [self.w[1,0],self.w[0,0]],
                [-self.w[0,0],self.w[1,0]]
            ]
        )/(np.linalg.norm(self.w)**2)
        self.a = -self.b*self.w/(np.linalg.norm(self.w)**2)

    def __call__(self, num_data:int, scaler:DataScaler=None) -> np.ndarray:
        base = 2*np.random.randn(num_data, 2)-1
        y = base[:,1]
        base[:,1] = np.where(y>0, y+1, y-1)
        X = np.array([(self.A @ x.reshape(-1,1)+self.a).reshape(-1) for x in base])
        if scaler is not None:
            self.scaler = scaler
            X = scaler(X)
        label = np.where(y>0, 1, 0)
        return X, label
    
    def plot(self, X, y, ax=plt, c = None):
        ax.scatter(X[:,0], X[:,1], c=c if c is not None else y)
        xrg=np.linspace(min(X[:,0]), max(X[:,0]))
        yrg=-self.w[0,0]/self.w[1,0]*xrg-self.b/self.w[1,0]
        ind1 = yrg<=max(X[:,1])
        ind2 = yrg>=min(X[:,1])
        ind = ind1*ind2
        ax.plot(xrg[ind], yrg[ind], 'k--')
        
        

class NotValidDataTypeError:
    pass