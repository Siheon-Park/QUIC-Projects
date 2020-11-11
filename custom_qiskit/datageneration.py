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

def iris_data(len:int=150, a:tuple=(0, 1, 2, 3)):
    dim = np.array([i in a for i in range(4)])
    # import some data to play with
    iris = datasets.load_iris()
    # Take the first two features. We could avoid this by using a two-dim dataset
    ind = np.random.choice(150, len, replace=False)
    blind = np.array([True if i in ind else False for i in range(150)])
    X = iris.data[blind, :] # pylint: disable=no-member
    X = X[:,dim]
    X = X-np.mean(X)
    X = X/np.linalg.norm(X, axis=1).reshape(-1, 1)
    y = iris.target[blind] # pylint: disable=no-member
    y = 2*np.ceil(y/2)-1
    Xt = iris.data[~blind, :] # pylint: disable=no-member
    Xt = Xt[:,dim]
    Xt = Xt-np.mean(Xt)
    Xt = Xt/np.linalg.norm(Xt, axis=1).reshape(-1, 1)
    yt = iris.target[~blind] # pylint: disable=no-member
    yt = 2*np.ceil(yt/2)-1
    return X, y, Xt, yt

if __name__ == "__main__":
    X, y = iris_data(10)
    print(X)