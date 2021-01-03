import numpy as np
from sklearn import datasets

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
