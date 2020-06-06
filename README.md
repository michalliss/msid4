# Introduction

This repositiry shows two different approaches to image classification for fashion-mnist dataset. First is an implementation of K Nearest Neighbours classifirer, and the second one is a simple variation of Convolutional Neural Network, inspired by VGG16 network. Furthermore, I tried to compare my results with already existing benchmarks.
### Dataset
Fashion-mnist is a dataset of Zalando's article images. It consists of 60000 training and 10000 test images (28x28 pixels each) with associated labels. There are 10 classes of labels.

My goal is to, using provided training data, build a model that will be able to succesfuly cllasify (most) images of clothing from the test data.
# Methods
## KNN
For this part, I used my implementation of K Nearest Neighbours classifier, which I had previously used for text analysis for my System Analysis and Decision Support Methods classes. There was only one problem - it used hamming distance as a distance metric, and it cannot be applied to non-binary data. The solution was simple - I used euclidean distance instead. 

<distance>

In order to make the computations faster I used cupy instead of numpy in some critical sections in code. Thanks to it, calculating distance matrix is really fast, especially if run on powerful GPU (eg. using Google Colab).

    def distance(X, X_train):
	    X = cp.asarray(X)
		X_train = cp.asarray(X_train)
		return cp.asnumpy(-2 * cp.dot(X, X_train.T) + cp.sum(X_train**2, axis=1) + cp.sum(X**2, axis=1)[:, cp.newaxis])

The results can be seen in Results section
## CNN
For CNN 
# Resources
[No loop euclidean distance](https://medium.com/@souaravdey/l2-distance-matrix-vectorization-trick-26aa3247ac6c)
