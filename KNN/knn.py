import numpy as np
import cupy as cp
from tensorflow import keras


def distance(X, X_train):
    X = cp.asarray(X)
    X_train = cp.asarray(X_train)
    return cp.asnumpy(-2 * cp.dot(X, X_train.T) + cp.sum(X_train**2, axis=1) + cp.sum(X**2, axis=1)[:, cp.newaxis])


def sort_train_labels_knn(Dist, y):
    N1, N2 = Dist.shape
    temp = cp.asnumpy(cp.argsort(Dist))
    Dist = Dist.astype('uint8')
    for i in range(N1):
        Dist[i] = y[temp[i]]
    return Dist


def p_y_x_knn(y, k):
    N1, N2 = y.shape
    M = int(y.max())
    res = np.empty(shape=(N1, M + 1))
    for i in range(N1):
        res[i] = np.bincount((y[i][:k]).astype(
            int), weights=None, minlength=(M + 1)) / k
    return res


def classification_error(p_y_x, y_true):
    N, M = p_y_x.shape

    res = 0
    for x in range(N):
        max = len(p_y_x[x]) - np.argmax(p_y_x[x][::-1]) - 1
        if max != y_true[x]:
            res += 1

    return res / N


def test_model(X_val, X_train, y_val, y_train, k_value):
    hm = distance(X_val, X_train)
    srt = sort_train_labels_knn(hm, y_train)

    error = classification_error(p_y_x_knn(srt, k_value), y_val)
    return error


def load_data():
    fashion_mnist = keras.datasets.fashion_mnist
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

    return x_train, y_train, x_test, y_test


def main():
    x_train, y_train, x_test, y_test = load_data()
    n_train = 60000
    n_test = 10000
    k_val = 5

    x_train = (np.array(x_train[:n_train]) /
               255.0).reshape(-1, 784).astype('float32')
    y_train = np.array(y_train[:n_train]).astype('uint8')
    x_test = (np.array(x_test[:n_test]) /
              255.0).reshape(-1, 784).astype('float32')
    y_test = np.array(y_test[:n_test]).astype('uint8')

    err = test_model(x_test, x_train, y_test, y_train, k_val)
    print("Error: " + str(err))
    print("Correct: " + str((1 - err)*100) + "%")


main()
