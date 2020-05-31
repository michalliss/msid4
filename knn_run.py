from mnist import MNIST
import knn
import numpy as np

def load_data():
    mndata = MNIST('data')
    x_train, y_train = mndata.load_training()
    x_test, y_test = mndata.load_testing()

    return x_train, y_train, x_test, y_test

def main():
    x_train, y_train, x_test, y_test = load_data()

    print(x_train[0])
    x_train = np.array(x_train)
    print(x_train.shape)
    y_train = np.array(y_train)
    x_test = np.array(x_test)
    print(x_test.shape)
    y_test = np.array(y_test)
    k_values = range(1, 99, 2)
    error_best, best_k, errors = knn.model_selection_knn(x_test, x_train, y_test, y_train, k_values)
    print('Najlepsze k: {num1} i najlepszy blad: {num2:.4f}'.format(num1=best_k, num2=error_best))

if __name__ == '__main__':
   main()