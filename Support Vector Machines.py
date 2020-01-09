#!/usr/bin/env python
# coding: utf-8

# In[55]:


import os
import gzip
import numpy as np
import pandas as pd
from keras.datasets import fashion_mnist
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn import metrics


# In[56]:


def load_mnist(path, kind='train'):
  
    labels_path = os.path.join(path, '%s-labels-idx1-ubyte.gz.cpgz'% kind)
    images_path = os.path.join(path,'%s-images-idx3-ubyte.gz.cpgz'% kind)

    with gzip.open(labels_path, 'rb') as lbpath:
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8,
                    offset=8)

    with gzip.open(images_path, 'rb') as imgpath:
        images = np.frombuffer(imgpath.read(), dtype=np.uint8,
                    offset=16).reshape(len(labels), 784)

    return images, labels


# In[ ]:


def plot_graph(y_train, y_val, y_test, accuracy_train, accuracy_validation, accuracy_test):
    
    plt.figure(num=1)
    plt.plot(accuracy_train/len(train_y), color="red",
             label="train_accuracy")
    plt.plot(accuracy_validation/len(val_y),
             color="green", label="validation_accuracy")
    plt.plot(accuracy_test/len(test_y), color="blue",
             label="test_accuracy")
    plt.xticks(range(9), iteration)
    plt.legend()
    plt.xlabel("C")
    plt.ylabel("Accuracy")
    plt.savefig('SVM_graph.jpg')


# In[57]:


def LinearSVM(train_x, train_y, val_x, val_y, test_x, test_y):

    accuracy_test = np.zeros(len(range(-4, 5)))
    accuracy_train = np.zeros(len(range(-4, 5)))
    accuracy_val = np.zeros(len(range(-4, 5)))
    
    
    number_of_iterations = []
    
    for i in np.arange(-4, 5, dtype=float):
        
        c = 10**i
        number_of_iterations.append(str(c))
        classifier = LinearSVC(C=c)
        classifier.fit(train_x, train_y)
        index = int(i+4)
        accuracy_train[index] = np.sum(train_y == classifier.predict(train_x))
        accuracy_val[index] = np.sum(val_y ==classifier.predict(val_x))
        accuracy_test[index] = np.sum(y_test == classifier.predict(test_x))
        
        
    plot_graph(train_x, train_y, val_x, val_y, test_x, test_y, train_accuracy, validation_accuracy, test_accuracy)
    
    function = validation_accuracy_val.argmax()-4
    C = int(function)
    best_value_c = 10**best
    
    best_element = [train_accuracy[C]/len(train_y), validation_accuracy[C]/len(val_y), test_accuracy[C]/len(test_y)]
    
    print('Best Value of C:', best_value_c)
    
    return best_value_c, best_element


# In[58]:


def confusion_matrix_svm(max_c, x_train, y_train, x_test, y_test):
    
    classifier = LinearSVC(C=max_c)
    classifier.fit(x_train, y_train)
    y_predict = classifier.predict(x_test)
    
    x = y_test == y_predict
    test_accuracy = np.sum(x)
    
    cm_test = metrics.confusion_matrix(y_test, y_predict)
    print('Testing accuracy:', test_accuracy/len(y_test))
    print('Confusion Matrix:', cm_test)
    return test_accuracy, cm_test


# In[59]:


def poly_kernel_svm(max_c, linearSVM, x_train, y_train, x_validation, y_validation, x_test, y_test):
    
    
    train_accuracy = np.zeros(4)
    train_accuracy[0] = linearSVM[0]
    validation_accuracy = np.zeros(4)
    validation_accuracy[0] = linearSVM[1]
    test_accuracy = np.zeros(4)
    test_accuracy[0] = linearSVM[2]
    SV_number = [0]
    
    degrees = [2, 3, 4]
    for d in degrees:
        classifier = SVC(kernel='poly', degree=d, C=max_c, gamma='auto')
        classifier.fit(x_train, y_train)
        
        train_accuracy[i-1] = np.sum(y_train ==classifier.predict(x_train))/len(y_train)
        validation_accuracy[i-1] = np.sum(y_validation == classifier.predict(x_validation))/len(y_validation)
        test_accuracy[i-1] = np.sum(y_test == classifier.predict(x_test))/len(y_test)
        SV_number.append(classifierclf.n_support_)

    result = {'train_accuracy': train_accuracy,'validation_accuracy': validation_accuracy,'test_accuracy': test_accuracy,'Number of Support Vectors': SV_number}
    
    final_test_accuracy = test_accuracy.argmax()+1
    
    print(result)
    
    print(test_accuracy.argmax()+1)
    
    return result, final_test_accuracy


# In[62]:


if __name__ == "__main__":
    
    X, Y = load_mnist('/Users/coraljain/Desktop/data/')
    x_test, y_test = load_mnist('/Users/coraljain/Desktop/data/', kind='t10k')
    images_validation, labels_validation = images[int(0.8*len(images)):],labels[int(0.8*len(labels)):]
    images_train, labels_train = images[:int(0.8*len(images))],labels[:int(0.8*len(labels))]
    
    print(len(images_train), len(labels_train), len(images_validation), len(labels_validation), len(images_test), len(labels_test))
    
    maxc, linearSVM = LinearSVM(images_train, labels_train,images_validation, labels_validation, images_test, labels_test)

    test_accuracy, confusion_matrix = confusion_matrix_svm(maxc, images_train, labels_train, images_test, labels_test)

    degree = poly_kernel_svm(maxc, linearSVM, images_train, labels_train, images_validation, labels_validation,images_test, labels_test)

