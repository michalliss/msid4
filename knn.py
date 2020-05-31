from time import sleep, time

import numpy as np


# import cupy as cp


def hamming_distance(X, X_train):
    N1, D = X.shape
    N2, D = X_train.shape
    ones1 = (np.ones(shape=X.shape) - X)
    ones2 = X.dot((np.ones(shape=X_train.shape) - X_train).T)
    return ones1 @ X_train.T + ones2


def sort_train_labels_knn(Dist, y):
    N1, N2 = Dist.shape
    res = np.zeros(shape=Dist.shape)
    print("1")
    temp = np.argsort(Dist, kind="mergesort")
    for i in range(N1):
        res[i] = y[temp[i]]
    print("2")
    return res


def p_y_x_knn(y, k):
    N1, N2 = y.shape
    M = int(y.max())
    res = np.empty(shape=(N1, M + 1))
    for i in range(N1):
        res[i] = np.bincount((y[i][:k]).astype(int), weights=None, minlength=(M + 1)) / k

    return res


def classification_error(p_y_x, y_true):
    N, M = p_y_x.shape

    res = 0
    for x in range(N):
        max = len(p_y_x[x]) - np.argmax(p_y_x[x][::-1]) - 1
        if max != y_true[x]:
            res += 1

    return res / N


def model_selection_knn(X_val, X_train, y_val, y_train, k_values):
    start = time()
    hm = hamming_distance(X_val, X_train)
    end = time()
    print("HM= " + str(end - start))

    start = time()
    srt = sort_train_labels_knn(hm, y_train)
    end = time()
    print("str= " + str(end - start))
    print(srt.nbytes)

    def fn(k):
        return classification_error(p_y_x_knn(srt, k), y_val)

    errors = [(fn(k_values[i]), i) for i in range(len(k_values))]
    best_err, best_index = min(errors, key=lambda t: t[0])
    return best_err, k_values[best_index], [i for i, j in errors]
