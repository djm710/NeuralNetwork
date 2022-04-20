import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def getImageData():
    data = pd.read_csv('mnist_train.csv')
    data = np.array(data)
    np.random.shuffle(data)
    n, m = data.shape
    data_train = data[0:1000].T
    Y_train = data_train[0]
    X_train = data_train[1:m]
    _,m_train = X_train.shape
    X_train = X_train / 255.
    X_train = X_train.T
    return X_train, Y_train
