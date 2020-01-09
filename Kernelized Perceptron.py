#!/usr/bin/env python
# coding: utf-8

# In[5]:


import os
import gzip
import numpy as np
import pandas as pd
from keras.datasets import fashion_mnist
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt


# In[6]:


def load_mnist(path, kind='train'):
    
    labels_path = os.path.join(path,
                    '%s-labels-idx1-ubyte.gz.cpgz'
                    % kind)
    images_path = os.path.join(path,
                    '%s-images-idx3-ubyte.gz.cpgz'
                    % kind)

    with gzip.open(labels_path, 'rb') as lbpath:
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8,offset=8)

    with gzip.open(images_path, 'rb') as imgpath:
        images = np.frombuffer(imgpath.read(), dtype=np.uint8,offset=16).reshape(len(labels), 784)

    return images, labels


# In[ ]:


def plot_graph(training_mistakes,train_accuracy, validation_accuracy test_accuracy, degree):

    plt.figure(num=2)
    plt.plot(training_mistakes, color="red", label="training mistakes")
    plt.legend()
    plt.xlabel("Number of iterations")
    plt.ylabel("Number of mistakes")
    plt.title("The Number of Training Mistakes")
    plt.savefig("Kernelized perceptron")
    print("train_accuracy:", train_accuracy*100, "%")
    print("validation_accuracy:", validation_accuracy*100, "%")
    print("test_accuracy:", test_accuracy*100, "%")
    print("training mistakes:", training_mistakes)


# In[10]:


def Kernelized_perceptron(train_x, train_y, val_x, val_y, test_x, test_y, degree):
    
    cal1 = train_x.size/len(train_x)
    cal2 = shape=(len(train_x),int(cal1))
    cal3 = (np.zeros(cal2)+255)
    train_x = train_x/cal3
    
    alpha = np.zeros(shape=(len(train_x), 10))
    mistakes_training = np.zeros(5)
    
    
    mistake_train = 0
    mistake_val = 0
    mistake_test = 0

    for i in range(5):
        for j in range(len(train_x)):
            x = train_x[i]
            y = train_y[i]
            
            
            y_cap = (np.dot(train_x, x)+1)**degree, alpha
            y_cap = np.dot(y_cap).argmax()

            if y_cap != y:
                mistakes_training[i] += 1
                alpha[j, y] += 1
                alpha[j, y_hat] -= 1
     
        
    w = np.dot(alpha.T, x_train)
    
    a_0 = [pre.argmax() for pre in np.dot(x_train, w.T)]
    a_1 = [pre.argmax() for pre in np.dot(x_validation, w.T)]
    a_2 = [pre.argmax() for pre in np.dot(x_train, w.T)] 

    train_accuracy = np.sum(a_0 == train_y)/len(train_y)
    validation_accuracy = np.sum(a_1 == val_y)/len(val_y)
    test_accuracy = np.sum(a_2 == test_y)/len(test_y)
    
    plot_graph(training_mistakes,train_accuracy, validation_accuracy, test_accuracy, degree)
    


# In[12]:


if __name__ == '__main__':
    
    X, Y = load_mnist('/Users/coraljain/Desktop/data/')
    x_test, y_test = load_mnist('/Users/coraljain/Desktop/data/', kind='t10k')
    
    x_train, x_val, y_train, y_val = train_test_split(X, Y, test_size=0.2)
    
    Kernelized_perceptron(x_train, y_train, x_val, y_val, x_test, y_test, 2) #Value of degree could be passed in argument as 2/3/4.


# In[ ]:




