# MNIST Flask App
![](https://github.com/sleepokay/mnist-flask-app/blob/master/media/screenshot.png)

A [Flask](http://flask.pocoo.org/) web app for handwritten digit recognition using a convolutional neural network. The model was trained on the MNIST dataset in [TensorFlow](https://www.tensorflow.org/) using the [Keras API](https://github.com/fchollet/keras).

CNN Architecture:
[32C3-BN]*2-[32C5S2-BN]-0.4d-[64C3-BN]*2-[64C5S2-BN]-0.4d-F-256D-10D

nCm -> n CNNs with size mXm
BN -> Batch Normalization
xd -> Dropout of x
F -> Flatten
xD -> Dense layer of x units

Epochs : 20

Jupyter Notebook:
