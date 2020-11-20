import numpy as np
from numpy.core.defchararray import replace
from sklearn import svm, datasets

def data_generation(dim, num, label, **kwargs):
    cov = kwargs.get('cov', np.eye(dim))
    mean = kwargs.get('mean', np.zeros(dim))
    data = np.random.multivariate_normal(mean, cov, size=num)
    labels = np.ones(num)*label
    return data, labels

def DataGeneration(dim, num_train, dist, cent, normalize=True):
    X1, y1 = data_generation(dim, int(np.ceil(num_train/4)), 1, mean=np.array([dist, 0]), cov=cent*np.eye(dim))
    X2, y2 = data_generation(dim, int(np.ceil(num_train/4)), 1, mean=np.array([-dist, 0]), cov=cent*np.eye(dim))
    X3, y3 = data_generation(dim, int(np.ceil(num_train/4)), -1, mean=np.array([0, dist]), cov=cent*np.eye(dim))
    X4, y4 = data_generation(dim, int(np.ceil(num_train/4)), -1, mean=np.array([0, -dist]), cov=cent*np.eye(dim))

    X_train = np.vstack((X1, X2, X3, X4))
    if normalize:
        X_train = np.array([x/np.linalg.norm(x) for x in X_train])
    y_train = np.hstack((y1, y2, y3, y4))
    return X_train, y_train

def sklearn_data(kind:str, num_data:int, a:tuple, labels:tuple, option:str=None):
    if kind == 'iris':
        data = datasets.load_iris()
    elif kind == 'wine':
        data = datasets.load_wine()
    else:
        raise NoSuchDataset
    full_num_data, full_dim = data.data.shape
    # assert
    assert len(a)<=full_dim
    for l in labels:
        assert l in np.unique(data.target)
    _temp =[t in labels for t in data.target]
    data_data = data.data[_temp, :]
    data_target = data.target[_temp]
    normalize=True if 'n' in option else False
    binary=True if 'b' in option else False
    dim = np.array([i in a for i in range(full_dim)])
    full_num_data, full_dim = data_data.shape
    assert num_data<=full_num_data
    ind = np.random.choice(full_num_data, num_data, replace=False)
    blind = np.array([True if i in ind else False for i in range(full_num_data)])
    X = data_data[blind, :] # pylint: disable=no-member
    X = X[:,dim]
    X = X-np.mean(X)
    y = data_target[blind] # pylint: disable=no-member
    Xt = data_data[~blind, :] # pylint: disable=no-member
    Xt = Xt[:,dim]
    Xt = Xt-np.mean(Xt)
    yt = data_target[~blind] # pylint: disable=no-member
    if normalize:
        X = X/np.linalg.norm(X, axis=1).reshape(-1, 1)
        Xt = Xt/np.linalg.norm(Xt, axis=1).reshape(-1, 1)
    if binary:
        y = 2*(y==min(labels))-1
        yt = 2*(yt==min(labels))-1
    return X, y, Xt, yt

class NoSuchDataset:
    pass
