#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd 
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.ensemble import AdaBoostClassifier
import matplotlib.pyplot as plt
from numpy.core.umath_tests import inner1d

f = open('output.txt', 'a+')

def read_files():
    train_data = pd.read_csv("income.train.txt",header=None)
    dev_data = pd.read_csv("income.dev.txt",header=None)
    test_data = pd.read_csv("income.test.txt",header=None)
    train_data.columns = ["age", "work_class", "education", "marital_status", "occupation","race", "sex", "hours", "country", "income"]
    dev_data.columns = ["age", "work_class", "education", "marital_status", "occupation","race", "sex", "hours", "country", "income"]
    test_data.columns = ["age", "work_class", "education", "marital_status", "occupation","race", "sex", "hours", "country", "income"]
    return train_data,dev_data,test_data

def preprocessing_sklearn(training_data,development_data,testing_data):
    train_label = training_data.iloc[:,-1]
    dev_label = development_data.iloc[:,-1]
    test_label = testing_data.iloc[:,-1]

    train_data = training_data.drop('income',1)
    dev_data = development_data.drop('income',1)
    test_data = testing_data.drop('income',1)

    combined_dataset = pd.concat([train_data,dev_data,test_data],keys=['train','dev','test'])
    one_hot_encoded_data  = pd.get_dummies(combined_dataset)
    train_data, dev_data, test_data = one_hot_encoded_data.xs('train'), one_hot_encoded_data.xs('dev'), one_hot_encoded_data.xs('test')
    
    return train_data,train_label,dev_data,dev_label,test_data,test_label

def boosting_based_on_tree_depth(train_data,train_label,test_data,test_label,depths,number_of_trees):
    accuracy_depth = []
    accuracies = {}
    for depth in depths:
        decision_tree = DecisionTreeClassifier(max_depth = depth,random_state=1)
        accuracy = {depth:{}}
        for number_of_tree in number_of_trees:
            model = AdaBoostClassifier(decision_tree,n_estimators=number_of_tree).fit(train_data, train_label)
            predicted = model.predict(test_data)
            accuracy_tree = accuracy_score(test_label,predicted)*100
            accuracy_depth.append(accuracy_tree)
        accuracy[depth] = accuracy_depth
        accuracy_depth = []
        accuracies.update(accuracy)
    return accuracies

def boosting_based_on_number_of_trees(train_data,train_label,test_data,test_label,depths,number_of_trees):
    accuracy_tree = []
    accuracies = {}
    for number_of_tree in number_of_trees:
        for depth in depths:
            decision_tree = DecisionTreeClassifier(max_depth = depth,random_state=1)
            accuracy = {number_of_tree:{}}
            model = AdaBoostClassifier(decision_tree,n_estimators=number_of_tree).fit(train_data, train_label)
            predicted = model.predict(test_data)
            accuracy_depth = accuracy_score(test_label,predicted)*100
            accuracy_tree.append(accuracy_depth)
        accuracy[number_of_tree] = accuracy_tree
        accuracy_tree = []
        accuracies.update(accuracy)
    return accuracies

def print_accuracies_depth(accuracies_depth,type_of_data,number_of_trees):
    print("%s Accuracies for Number of boosting iterations vs Depth of Tree: Boosting" %(type_of_data),file = f)
    df = pd.DataFrame(accuracies_depth).T
    df.columns = number_of_trees
    print(df,file = f)
    print(file = f)

def print_accuracies_number_of_trees(accuracies_tree_number,type_of_data,depths):
    print("%s Accuracies for Depth of Tree vs Number of boosting iterations: Boosting" %(type_of_data),file = f)
    df = pd.DataFrame(accuracies_tree_number).T
    df.columns = depths
    print(df,file = f)
    print(file = f)

def plot_curve_depth(train_accuracies,dev_accuracies,test_accuracies,depths,number_of_trees):
    x_values = number_of_trees
    for depth_of_tree in depths: 
        y_value_train =  train_accuracies[depth_of_tree]
        y_value_dev = dev_accuracies[depth_of_tree]
        y_value_test = test_accuracies[depth_of_tree]
        plt.plot(x_values, y_value_train, color = 'red', marker='o', linestyle='solid', label='Training Data')
        plt.plot(x_values, y_value_dev, color = 'green', marker='o', linestyle='solid', label='Development Data')
        plt.plot(x_values, y_value_test, color = 'blue', marker='o', linestyle='solid', label='Testing Data')
        plt.legend(loc='upper left')
        plt.xlabel("Number of Boosting Iterations")
        plt.ylabel("Accuracy (%)")
        plt.title("Depth of Tree %d vs Number of Boosting Iterations: Boosting" %(depth_of_tree))
        save_file_name = "Depth_of_Tree_%d_vs_Number_of_Boosting_Iterations_Boosting.png" %(depth_of_tree)
        plt.savefig(save_file_name)
        y_value_train = []
        y_value_dev = []
        y_value_test = []
        plt.clf() 
        #plt.show()

def plot_curve_number_of_trees(train_accuracies,dev_accuracies,test_accuracies,depths,number_of_trees):
    x_values = depths
    for number_of_tree in number_of_trees: 
        y_value_train =  train_accuracies[number_of_tree]
        y_value_dev = dev_accuracies[number_of_tree]
        y_value_test = test_accuracies[number_of_tree]
        plt.plot(x_values, y_value_train, color = 'red', marker='o', linestyle='solid', label='Training Data')
        plt.plot(x_values, y_value_dev, color = 'green', marker='o', linestyle='solid', label='Development Data')
        plt.plot(x_values, y_value_test, color = 'blue', marker='o', linestyle='solid', label='Testing Data')
        plt.legend(loc='upper left')
        plt.xlabel("Depth of tree")
        plt.ylabel("Accuracy (%)")
        plt.title("Boosting Iteration %d vs Depth of Tree: Boosting" %(number_of_tree))
        save_file_name = "Boosting_Iteration_%d_vs_Depth_of_Tree_Boosting.png" %(number_of_tree)
        plt.savefig(save_file_name)
        y_value_train = []
        y_value_dev = []
        y_value_test = []
        plt.clf() 
        #plt.show()

def main():
    train_data,dev_data,test_data = read_files()
    train_data_sk,train_label_sk,dev_data_sk,dev_label_sk,test_data_sk,test_label_sk = preprocessing_sklearn(train_data,dev_data,test_data)
    depths = [1,2,3,5,10]
    number_of_trees = [10,20,40,60,80,100]
    print("Boosting:",file = f)
    print(file = f)
    
    train_accuracies_depth = boosting_based_on_tree_depth(train_data_sk,train_label_sk,train_data_sk,train_label_sk,depths,number_of_trees)
    print_accuracies_depth(train_accuracies_depth,"Train",number_of_trees)
    dev_accuracies_depth = boosting_based_on_tree_depth(train_data_sk,train_label_sk,dev_data_sk,dev_label_sk,depths,number_of_trees)
    print_accuracies_depth(dev_accuracies_depth,"Development",number_of_trees)
    test_accuracies_depth = boosting_based_on_tree_depth(train_data_sk,train_label_sk,test_data_sk,test_label_sk,depths,number_of_trees)
    print_accuracies_depth(test_accuracies_depth,"Testing",number_of_trees)
    plot_curve_depth(train_accuracies_depth,dev_accuracies_depth,test_accuracies_depth,depths,number_of_trees)

    train_accuracies_tree_number = boosting_based_on_number_of_trees(train_data_sk,train_label_sk,train_data_sk,train_label_sk,depths,number_of_trees)
    print_accuracies_number_of_trees(train_accuracies_tree_number,"Train",depths)
    dev_accuracies_tree_number = boosting_based_on_number_of_trees(train_data_sk,train_label_sk,dev_data_sk,dev_label_sk,depths,number_of_trees)
    print_accuracies_number_of_trees(dev_accuracies_tree_number,"Development",depths)
    test_accuracies_tree_number = boosting_based_on_number_of_trees(train_data_sk,train_label_sk,test_data_sk,test_label_sk,depths,number_of_trees)
    print_accuracies_number_of_trees(test_accuracies_tree_number,"Testing",depths)
    plot_curve_number_of_trees(train_accuracies_tree_number,dev_accuracies_tree_number,test_accuracies_tree_number,depths,number_of_trees)

main()

f.close()

