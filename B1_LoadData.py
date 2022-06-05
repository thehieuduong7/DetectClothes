import tensorflow as tf
from tensorflow.keras import datasets
import cv2
# pip install tensorflow
# pip install opencv-python
# data: shuffled and split between train and test sets
(X_train, y_train), (X_test, y_test) = datasets.cifar10.load_data()

sample = X_test[1000]
cv2.imshow('digit', sample)
cv2.waitKey(0)
