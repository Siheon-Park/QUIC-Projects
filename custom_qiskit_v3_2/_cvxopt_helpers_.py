import numpy as np
import cvxopt

class _Matrix_Helper:

    @staticmethod
    def __find_REDUCED_kernel_matrix__(svm, data, polary):
        n = svm.num_data
        P = np.empty((n, n))
        for i in range(n):
            for j in range(n):
                P[i, j] = (1/svm.k+svm.kernel(data[i], data[j]))*polary[i]*polary[j]
        return cvxopt.matrix(P, (n, n), 'd') 

    @staticmethod
    def __find_kernel_matrix__(svm, data, polary):
        n = svm.num_data
        P = np.empty((n, n))
        for i in range(n):
            for j in range(n):
                P[i, j] = svm.kernel(data[i], data[j])*polary[i]*polary[j]
        return cvxopt.matrix(P, (n, n), 'd') 

    @staticmethod
    def __find_matrix__REDUCED_SVM__(P, svm):
        n = svm.num_data
        q = cvxopt.matrix(-1.0, (n, 1), 'd')
        if svm.C is None:
            h = cvxopt.matrix(0.0, (n,1), 'd')
            G = cvxopt.matrix(-np.eye(n), (n,n), 'd')
        else:
            G1 = np.eye(n)
            G2 = -np.eye(n)
            h1 = svm.C*np.ones(n)
            h2 = np.zeros(n)
            h = cvxopt.matrix(np.concatenate([h1, h2]), (2*n, 1), 'd')
            G = cvxopt.matrix(np.vstack([G1, G2]), (2*n, n), 'd')
        return P, q, G, h, None, None

    @staticmethod
    def __find_matrix__REDUCED_QASVM__(P, svm):
        n = svm.num_data
        A = cvxopt.matrix(1.0, (1, n), 'd')
        h = cvxopt.matrix(0.0, (n, 1), 'd')
        G = cvxopt.matrix(-np.eye(n), (n,n), 'd')
        if svm.C is None:
            b = cvxopt.matrix(1.0, (1, 1), 'd')
            q = cvxopt.matrix(0.0, (n, 1), 'd')
        else:
            b = cvxopt.matrix(svm.C, (1, 1), 'd')
            q = cvxopt.matrix(-1.0, (n, 1), 'd')
        return P, q, G, h, A, b

    @staticmethod
    def __find_matrix__SVM__(P, svm):
        n = svm.num_data
        q = cvxopt.matrix(-1.0, (n, 1), 'd')
        A = cvxopt.matrix(svm.polary, (1, n), 'd')
        b = cvxopt.matrix(0.0, (1, 1), 'd')
        if svm.C is None:
            h = cvxopt.matrix(0.0, (n,1), 'd')
            G = cvxopt.matrix(-np.eye(n), (n,n), 'd')
        else:
            G1 = np.eye(n)
            G2 = -np.eye(n)
            h1 = svm.C*np.ones(n)
            h2 = np.zeros(n)
            h = cvxopt.matrix(np.concatenate([h1, h2]), (2*n, 1), 'd')
            G = cvxopt.matrix(np.vstack([G1, G2]), (2*n, n), 'd')
        return P, q, G, h, A, b

    @staticmethod
    def __find_matrix__QASVM__(P, svm):
        n = svm.num_data
        q = cvxopt.matrix(-1.0, (n, 1), 'd')
        A1 = svm.polary
        A2 = np.ones(n) if svm.C is None else svm.C*np.ones(n)
        A = cvxopt.matrix(np.vstack([A1, A2]), (2, n), 'd')
        b = cvxopt.matrix([0.0, 1.0], (2, 1), 'd')
        h = cvxopt.matrix(0.0, (n,1), 'd')
        G = cvxopt.matrix(-np.eye(n), (n,n), 'd')
        return P, q, G, h, A, b

    @staticmethod
    def __find_matrix__REDUCED_PRIMAL_SVM__(P, svm):
        n = svm.num_data
        q1 = np.zeros(n)
        if svm.C is None:
            q2 = np.zeros(n)
        else:
            q2 = svm.C*np.ones(n)
        q = cvxopt.matrix(np.concatenate((q1, q2)), (2*n, 1), 'd')

        P01 = np.zeros((n, n))
        P10 = np.zeros((n, n))
        P11 = np.zeros((n, n))
        P0 = np.hstack((P, P01))
        P1 = np.hstack((P10, P11))
        PP = cvxopt.matrix(np.vstack((P0, P1)), (2*n, 2*n), 'd')

        G00 = np.zeros((n, n))
        G10 = -P
        G20 = -np.eye(n)
        G30 = np.eye(n)
        if svm.C is None:
            G01 = np.zeros((n,n))
            G11 = np.zeros((n,n))
        else:
            G01 = -np.eye(n)
            G11 = -np.eye(n)
        G21 = np.zeros((n, n))
        G31 = np.zeros((n, n))
        G0 = np.hstack((G00, G01))
        G1 = np.hstack((G10, G11))
        G2 = np.hstack((G20, G21))
        G3 = np.hstack((G30, G31))
        if svm.C is None:
            G = cvxopt.matrix(np.vstack((G0, G1, G2)), (3*n, 2*n), 'd')
        else:
            G = cvxopt.matrix(np.vstack((G0, G1, G2, G3)), (4*n, 2*n), 'd')

        h1 = np.zeros(n)
        h2 = -np.ones(n)
        h3 = np.zeros(n)
        if svm.C is None:
            h = cvxopt.matrix(np.concatenate((h1, h2, h3)), (3*n, 1), 'd')
        else:
            h4 = svm.C*np.ones(n)
            h = cvxopt.matrix(np.concatenate((h1, h2, h3, h4)), (4*n, 1), 'd')

        return PP, q, G, h, None, None

    @staticmethod
    def __find_matrix__REDUCED_PRIMAL_QASVM__(P, svm):
        n = svm.num_data
        q1 = np.zeros(n)
        if svm.C is None:
            q2 = np.zeros(n)
        else:
            q2 = svm.C*np.ones(n)
        q = cvxopt.matrix(np.concatenate((q1, q2)), (2*n, 1), 'd')

        P01 = np.zeros((n, n))
        P10 = np.zeros((n, n))
        P11 = np.zeros((n, n))
        P0 = np.hstack((P, P01))
        P1 = np.hstack((P10, P11))
        PP = cvxopt.matrix(np.vstack((P0, P1)), (2*n, 2*n), 'd')

        G00 = np.zeros((n, n))
        G10 = -P
        G20 = -np.eye(n)
        if svm.C is None:
            G01 = np.zeros((n,n))
            G11 = np.zeros((n,n))
        else:
            G01 = -np.eye(n)
            G11 = -np.eye(n)
        G21 = np.zeros((n, n))
        G0 = np.hstack((G00, G01))
        G1 = np.hstack((G10, G11))
        G2 = np.hstack((G20, G21))
        G = cvxopt.matrix(np.vstack((G0, G1, G2)), (3*n, 2*n), 'd')

        h1 = np.zeros(n)
        h2 = -np.ones(n)
        h3 = np.zeros(n)
        h = cvxopt.matrix(np.concatenate((h1, h2, h3)), (3*n, 1), 'd')

        A1 = np.ones(n)
        A2 = np.zeros(n)
        A = cvxopt.matrix(np.concatenate((A1, A2)), (1, 2*n), 'd')

        if svm.C is None:
            b = cvxopt.matrix(1.0, (1, 1), 'd')
        else:
            b = cvxopt.matrix(svm.C, (1, 1), 'd')

        return PP, q, G, h, A, b
