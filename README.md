# MNIST Flask App
 
A [Flask](http://flask.pocoo.org/) web app for handwritten digit recognition using a CNN architecture. This model is trained on the dataset from [Kaggle](https://www.kaggle.com/c/digit-recognizer/data) in Tensorflow. Got an accuracy of ``0.99614`` in this contest.

CNN Architecture:

[32C3-BN]*2-[32C5S2-BN]-0.4d-[64C3-BN]*2-[64C5S2-BN]-0.4d-F-256D-10D

nCm -> n CNNs with size mXm  
BN -> Batch Normalization  
xd -> Dropout of x  
F -> Flatten  
xD -> Dense layer of x units  

Epochs : 20

[Jupyter Notebook](https://github.com/karthikmuru/mnist-flask-app/blob/master/MNIST.ipynb)