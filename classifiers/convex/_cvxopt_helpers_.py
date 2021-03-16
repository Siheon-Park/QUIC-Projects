
import numpy as np
import cvxopt

class _Matrix_Helper:
    @staticmethod
    def __find_kernel_matrix__(data, polary, kernel):
        #K = pairwise_kernels(data, data, metric=kernel)
        K = np.empty((len(data), len(data)))
        for i in range(len(data)):
            for j in range(len(data)):
                K[i, j] = kernel(data[i], data[j])
                """
        if centering:
            kl = KernelCenterer()
            K = kl.fit_transform(K)
            """
        Y = polary.reshape(-1,1)@polary.reshape(1,-1)
        return K, Y

    @staticmethod
    def __find_matrix__REDUCED_SVM__(P, n, C, polary):
        q = cvxopt.matrix(-1.0, (n, 1), 'd')
        if C is None:
            h = cvxopt.matrix(0.0, (n,1), 'd')
            G = cvxopt.matrix(-np.eye(n), (n,n), 'd')
        else:
            G1 = np.eye(n)
            G2 = -np.eye(n)
            h1 = C*np.ones(n)
            h2 = np.zeros(n)
            h = cvxopt.matrix(np.concatenate([h1, h2]), (2*n, 1), 'd')
            G = cvxopt.matrix(np.vstack([G1, G2]), (2*n, n), 'd')
        return P, q, G, h, None, None

    @staticmethod
    def __find_matrix__REDUCED_QASVM__(P, n, C, polary):
        A = cvxopt.matrix(1.0, (1, n), 'd')
        h = cvxopt.matrix(0.0, (n, 1), 'd')
        G = cvxopt.matrix(-np.eye(n), (n,n), 'd')
        if C is None:
            b = cvxopt.matrix(1.0, (1, 1), 'd')
            q = cvxopt.matrix(0.0, (n, 1), 'd')
        else:
            b = cvxopt.matrix(C, (1, 1), 'd')
            q = cvxopt.matrix(-1.0, (n, 1), 'd')
        return P, q, G, h, A, b

    @staticmethod
    def __find_matrix__SVM__(P, n, C, polary):
        q = cvxopt.matrix(-1.0, (n, 1), 'd')
        A = cvxopt.matrix(polary, (1, n), 'd')
        b = cvxopt.matrix(0.0, (1, 1), 'd')
        if C is None:
            h = cvxopt.matrix(0.0, (n,1), 'd')
            G = cvxopt.matrix(-np.eye(n), (n,n), 'd')
        else:
            G1 = np.eye(n)
            G2 = -np.eye(n)
            h1 = C*np.ones(n)
            h2 = np.zeros(n)
            h = cvxopt.matrix(np.concatenate([h1, h2]), (2*n, 1), 'd')
            G = cvxopt.matrix(np.vstack([G1, G2]), (2*n, n), 'd')
        return P, q, G, h, A, b

    @staticmethod
    def __find_matrix__QASVM__(P, n, C, polary):
        A1 = polary
        A2 = np.ones(n)
        A = cvxopt.matrix(np.vstack([A1, A2]), (2, n), 'd')
        h = cvxopt.matrix(0.0, (n,1), 'd')
        G = cvxopt.matrix(-np.eye(n), (n,n), 'd')
        if C is None:
            b = cvxopt.matrix([0.0, 1.0], (2, 1), 'd')
            q = cvxopt.matrix(0.0, (n, 1), 'd')
        else:
            b = cvxopt.matrix([0.0, C], (2, 1), 'd')
            q = cvxopt.matrix(-1.0, (n, 1), 'd')
        return P, q, G, h, A, b

    @staticmethod
    def __find_matrix__REDUCED_PRIMAL_SVM__(P, n, C, polary):
        q1 = np.zeros(n)
        if C is None:
            q2 = np.zeros(n)
        else:
            q2 = C*np.ones(n)
        q = cvxopt.matrix(np.concatenate((q1, q2)), (2*n, 1), 'd')

        P01 = np.zeros((n, n))
        P10 = np.zeros((n, n))
        P11 = np.zeros((n, n))
        P0 = np.hstack((P, P01))
        P1 = np.hstack((P10, P11))
        PP = cvxopt.matrix(np.vstack((P0, P1)), (2*n, 2*n), 'd')

        G00 = np.zeros((n, n))
        G01 = -np.eye(n)
        G10 = -P
        G11 = -np.eye(n)
        G20 = -np.eye(n)
        G21 = np.zeros((n, n))
        G0 = np.hstack((G00, G01))
        G1 = np.hstack((G10, G11))
        G2 = np.hstack((G20, G21))
        h0 = np.zeros((n, 1))
        h1 = -np.ones((n, 1))
        h2 = np.zeros((n, 1))
        if C is None:
            G = cvxopt.matrix(np.vstack((G0, G1, G2)), (3*n, 2*n), 'd')
            h = cvxopt.matrix(np.vstack((h0, h1, h2)), (3*n, 1), 'd')
        else:
            G30 = np.eye(n)
            G31 = np.zeros((n, n))
            G3 = np.hstack((G30, G31))
            h3 = C*np.ones((n, 1))
            G = cvxopt.matrix(np.vstack((G0, G1, G2, G3)), (4*n, 2*n), 'd')
            h = cvxopt.matrix(np.vstack((h0, h1, h2, h3)), (4*n, 1), 'd')
        return PP, q, G, h, None, None

    @staticmethod
    def __find_matrix__REDUCED_PRIMAL_QASVM__(P, n, C, polary):
        assert C is not None
        q1 = np.zeros(n)
        q2 = np.ones(n)
        q = cvxopt.matrix(np.concatenate((q1, q2)), (2*n, 1), 'd')

        P01 = np.zeros((n, n))
        P10 = np.zeros((n, n))
        P11 = np.zeros((n, n))
        P0 = np.hstack((P, P01))
        P1 = np.hstack((P10, P11))
        PP = cvxopt.matrix(np.vstack((P0, P1)), (2*n, 2*n), 'd')

        G00 = np.zeros((n, n))
        G01 = -np.eye(n)
        G10 = -P
        G11 = -np.eye(n)
        G20 = -np.eye(n)
        G21 = np.zeros((n, n))
        G0 = np.hstack((G00, G01))
        G1 = np.hstack((G10, G11))
        G2 = np.hstack((G20, G21))
        h0 = np.zeros((n, 1))
        h1 = -np.ones((n, 1))/C
        h2 = np.zeros((n, 1))
        G = cvxopt.matrix(np.vstack((G0, G1, G2)), (3*n, 2*n), 'd')
        h = cvxopt.matrix(np.vstack((h0, h1, h2)), (3*n, 1), 'd')
        A0 = np.ones((1, n))
        A1 = np.zeros((1, n))
        A = cvxopt.matrix(np.hstack((A0, A1)), (1, 2*n), 'd')
        b = cvxopt.matrix(1.0, (1,1), 'd')

        return PP, q, G, h, A, b
