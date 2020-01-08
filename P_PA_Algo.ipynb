#!/usr/bin/env python
# coding: utf-8

# In[630]:


import numpy as np
import pandas as pd
from keras.utils import to_categorical
from keras.datasets import fashion_mnist
import matplotlib.pyplot as plt


# In[631]:


output_file = open('output_file.txt', 'w')


# In[632]:


#Preprocessing for train and test data
def pre_processing(images,labels,isMultiple=False):
    rows = 28
    columns = 28
    number_class_labels = len(np.unique(labels))
    images = images.reshape(images.shape[0], rows, columns, 1)
    images = images.astype('float32')
    images = images/255
    if isMultiple:
        labels = to_categorical(labels, number_class_labels)
    return images,labels


# In[633]:


#Binary classification to classify even labels (0, 2, 4, 6, 8) and odd labels (1, 3, 5, 7, 9)
def binary_classification(y_label):
    y_update = []
    for y in y_label:
        if y%2 == 0:
            y_update.append(1)
        else:
            y_update.append(-1)
    return y_update


# In[634]:


#Initialising vector to zero
def vectors_initialisation(zero_vector):
    v = np.zeros((zero_vector[0].shape))
    return v


# In[635]:


#Calculation of learning rate for Passive Aggressive (for Perceptron this value is 0)
def calculate_learning_rate(fvector, cls_label, weights):
    return (1 - cls_label * np.dot(weights, fvector)) / (np.square(np.linalg.norm(fvector)))


# In[636]:


#Calculation of test and train accuracy for Perceptron and Passive Aggressive
def accuracy_calculation(mistake,examples):

    accuracy = (1 - mistake/examples)*100
    return accuracy


# In[637]:


#Learning curves for Perceptron & Passive Aggressive
def learning_curve(mistakes,iterations,name_of_algo,file_name):
    plt.plot(iterations, mistakes, color = 'g', marker='o', linestyle='solid')
    plt.xlabel('Number of Iterations')
    plt.ylabel('Number of Mistakes')
    plt.title("Learning curve: %s" %(name_of_algo))
    plt.savefig(file_name)
    plt.show()


# In[638]:


#Accuracy curves for Perceptron & Passive Aggressive
def accuracy_curve(x_values,accuracies,x_label,name_of_algo,file_name):
    plt.plot(x_values, accuracies, color = 'r', marker='o', linestyle='solid')
    plt.xlabel(x_label)
    plt.ylabel('Accuracy (%)')
    plt.title("Accuracy curve: %s" %(name_of_algo))
    plt.savefig(file_name)
    plt.show()


# In[639]:


#Perceptron General Learning Curve plot
def perceptron_general_learning_curve(x_train,y_train,x_test,y_test, save_file_name):
    
    w = vectors_initialisation(x_train)
    w = np.squeeze(np.asarray(w))
    w = np.ravel(w)
    
    mistakes = []
    iterations = []
    examples = x_train.shape[0]
    train_data_accuracies = []
    test_data_accuracies = []
    row_range = [5000,10000,15000,20000,25000]

    for rows in row_range:
        print(file=output_file)
        x_train_subset = x_train[:rows,:]
        y_train_subset = y_train[:rows]
        train_data_accuracy,test_data_accuracy = perceptron(x_train_subset, y_train_subset, x_test,y_test, 5,"Do not plot graph")
        train_data_accuracies.append(train_data_accuracy)
        test_data_accuracies.append(test_data_accuracy)
        
    plt.plot(row_range, train_data_accuracies, color = 'b', marker='o', linestyle='solid', label='Training Data')
    plt.plot(row_range, test_data_accuracies, color = 'r', marker='o', linestyle='solid', label='Testing Data')
    plt.legend(loc='upper left')
    plt.xlabel("Number of Examples")
    plt.ylabel("Accuracy (%)")
    plt.title("General learning curve: Perceptron Algorithm")
    plt.savefig(save_file_name)
    plt.show()


# In[640]:


#Passive Aggressive General Learning Curve plot
def passive_aggressive_general_learning_curve(x_train,y_train,x_test,y_test,save_file_name):
    
    w = vectors_initialisation(x_train)
    w = np.squeeze(np.asarray(w))
    w = np.ravel(w)
    mistakes = []
    iterations = []
    examples = x_train.shape[0]
    train_data_accuracies = []
    test_data_accuracies = []
    row_range = [5000,10000,15000,20000,25000]

    for rows in row_range:
        print(file=output_file)
        x_train_subset = x_train[:rows,:]
        y_train_subset = y_train[:rows]
        train_data_accuracy,test_data_accuracy = passive_aggressive(x_train_subset, y_train_subset, x_test,y_test, 5,"Do not plot graph")
        train_data_accuracies.append(train_data_accuracy)
        test_data_accuracies.append(test_data_accuracy)
    
    plt.plot(row_range, train_data_accuracies, color = 'r', marker='o', linestyle='solid', label='Training Data')
    plt.plot(row_range, test_data_accuracies, color = 'b', marker='o', linestyle='solid', label='Testing Data')
    plt.legend(loc='upper left')
    plt.xlabel("Number of Examples")
    plt.ylabel("Accuracy (%)")
    plt.title("General learning curve: Passive Aggressive Algorithm")
    plt.savefig(save_file_name)
    plt.show()


# In[641]:


#Perceptron Algorithm 
def perceptron(x_train,y_train,x_test,y_test,maxiter, a):
    w = vectors_initialisation(x_train)
    w = np.squeeze(np.asarray(w))
    w = np.ravel(w)
    
    mistakes = []
    iterations = []
    examples = x_train.shape[0]
    
    train_data_accuracies = []
    test_data_accuracies = []
    
    learning_rate = 1
    
    for i in range(0,maxiter):
        mistake = 0
        for j in range(examples):
            x = np.squeeze(np.asarray(x_train[j]))
            x = np.ravel(x)
            
            y_cap = np.sign(np.dot(x,w))
            if y_cap == 0:
                y_cap = -1
            if y_cap != y_train[j]:
                mistake = mistake + 1
                w = w + learning_rate * np.dot(y_train[j],x)
     
        mistakes.append(mistake)
        iterations.append(i + 1)

        train_data_accuracy = accuracy_calculation(mistake,examples)
        test_data_accuracy = test_algo(x_test,y_test,w,0)
        
        train_data_accuracies.append(train_data_accuracy)
        test_data_accuracies.append(test_data_accuracy)

    #Plotting learning and accuracy curves when required    
    if a == "plot graph":
        
        learning_curve(mistakes,iterations,"Perceptron","Learning_curve_Perceptron.png")
        accuracy_curve(iterations,train_data_accuracies,"Number of Iterations","Training Perceptron","Perceptron_Accuracy_Curve_Training.png")
        accuracy_curve(iterations,test_data_accuracies,"Number of Iterations","Testing Perceptron","Perceptron_Accuracy_Testing_Curve.png")

    train_accuracy = accuracy_calculation(mistake, examples)
    test_accuracy = test_algo(x_test,y_test,w,0)
    
    print(file=output_file)
    print("Training Accuracy = %0.2f" %(train_accuracy), file=output_file)
    print("Test Accuracy = %0.2f" %(test_accuracy), file=output_file)
    print(file=output_file)

    return train_accuracy, test_accuracy


# In[642]:


#Passive Aggressive Algorithm
def passive_aggressive(x_train,y_train,x_test,y_test,maxiter,a):
   # time_stamp("Passive Aggressive Algorithm")
    w = vectors_initialisation(x_train)
    w = np.squeeze(np.asarray(w))
    w = np.ravel(w)
    examples = x_train.shape[0]
    
    mistakes = []
    iterations = []
    
    
    train_data_accuracies = []
    test_data_accuracies = []

    for i in range(0,maxiter):
        mistake = 0
        for j in range(examples):
            x = np.squeeze(np.asarray(x_train[j]))
            x = np.ravel(x)
            y_cap = np.sign(np.dot(w,x))
            if y_cap == 0:
                y_cap = -1
            if y_train[j] != y_cap:
                learning_rate = calculate_learning_rate(x,y_train[j],w)
                mistake = mistake + 1
                w = w + learning_rate * np.dot(y_train[j],x)

        mistakes.append(mistake)
        iterations.append(i + 1)
        
        train_data_accuracy = accuracy_calculation(mistake,examples)
        test_data_accuracy = test_algo(x_test,y_test,w,0)
        train_data_accuracies.append(train_data_accuracy)
        test_data_accuracies.append(test_data_accuracy)
    
    
    #Plotting learning and accuracy curves when required
    if a == "plot graph":
        learning_curve(mistakes,iterations,"Passive Aggressive","Learning_Curve_Passive_Aggressive.png")
        accuracy_curve(iterations,train_data_accuracies,"Number of Iterations","Training Passive Aggressive","Passive_Aggressive_Training_Accuracy_Curve.png")
        accuracy_curve(iterations,test_data_accuracies,"Number of Iterations","Testing Passive Aggressive","Passive_Aggressive_Testing_Accuracy_Curve.png")

    train_accuracy = accuracy_calculation(mistake,examples)
    test_accuracy = test_algo(x_test,y_test,w,0)

    print(file=output_file)
    print("Training Accuracy = %0.2f" %(train_accuracy), file=output_file)
    print("Test Accuracy = %0.2f" %(test_accuracy), file=output_file)
    print(file=output_file)

    return train_accuracy, test_accuracy


# In[643]:


#Testing Algorithm for Perceptron and Passive Aggresive
def test_algo(fvector,cls_label,weights,bias):
    
    examples = fvector.shape[0]
    
    mistakes = 0
    
    
    for j in range(examples):
        x_sq = np.squeeze(np.asarray(fvector[j]))
        x = np.ravel(x_sq)
        
        y_cap = np.sign(np.dot(x,weights)+bias)
        
        if y_cap == 0:
            y_cap = -1
        if y_cap != cls_label[j]:
            mistakes = mistakes + 1
            
    acc = accuracy_calculation(mistakes, examples)
    return acc


# In[644]:


#Average Perceptron Algorithm
def average_perceptron(x_train,y_train,x_test,y_test,maxiter):
    
    learning_rate = 1
    
    w = vectors_initialisation(x_train)
    w = np.squeeze(np.asarray(w))
    w = np.ravel(w)
    
    
    examples = x_train.shape[0]
    weighted_sum = 0
    #start_time = dt.datetime.now()

    for i in range(0, maxiter):
        mistakes = 0
        for j in range(examples):
            x = np.squeeze(np.asarray(x_train[j]))
            x = np.ravel(x)
            y_cap = np.sign(np.dot(x,w))
            
            if y_cap == 0:
                y_cap = -1
            if y_cap != y_train[j]:
                mistakes = mistakes + 1
                w = w + learning_rate * np.dot(y_train[j],x)
                weighted_sum = w + weighted_sum
                
    weighted_avg = weighted_sum/examples

    print(file=output_file)
    print("Accuracy of training data= %0.2f" %(accuracy_calculation(mistakes,examples)))
    print("Accuracy of testing data = %0.2f" %(test_algo(x_test,y_test,weighted_avg,0)))
    print(file=output_file)


# In[645]:


#Extraction of test and train data from Fashion MNIST dataset
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()


# In[646]:


#Collection of Binary classified and Preprocessed data
y_train_class = binary_classification(y_train)
y_test_class = binary_classification(y_test)
x_train_pr, y_train_pr = pre_processing(x_train,y_train_class)
x_test_pr, y_test_pr = pre_processing(x_test,y_test_class)


# In[652]:


#Binary Classification using Perceptron and plotting its curves
perceptron(x_train_pr, y_train_pr,x_test_pr, y_test_pr,20, "plot graph")


# In[654]:


#Binary Classification using Passive Aggressive and plotting its curves
passive_aggressive(x_train_pr, y_train_pr,x_test_pr, y_test_pr,20, "plot graph")


# In[649]:


#Classification using Average Perceptron and calculation of accuracy
average_perceptron(x_train_pr,y_train_pr,x_test_pr,y_test_pr,50)


# In[650]:


#General curve plot of perceptron
perceptron_general_learning_curve(x_train_pr,y_train_pr,x_test_pr,y_test_pr, "General learning curve- Perceptron")


# In[651]:


#General curve plot of passive aggressive
passive_aggressive_general_learning_curve(x_train_pr,y_train_pr,x_test_pr,y_test_pr, "General learning curve- PA")


# In[ ]:




