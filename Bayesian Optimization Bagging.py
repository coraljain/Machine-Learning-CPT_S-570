#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import logging
import numpy as np
from sklearn import svm, datasets
from sklearn.model_selection import cross_val_score
import pandas as pd
import random
import seaborn as sns
import time
from numpy import mean
from numpy import std
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from keras.datasets import fashion_mnist
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.optimizers import SGD


# In[ ]:


# load train and test dataset
def load_dataset():
    # load dataset
    (trainX, trainY), (testX, testY) = fashion_mnist.load_data()
    # reshape dataset to have a single channel
    trainX = trainX.reshape((trainX.shape[0], 28, 28, 1))
    testX = testX.reshape((testX.shape[0], 28, 28, 1))
    # one hot encode target values
    trainY = to_categorical(trainY)
    testY = to_categorical(testY)
    return trainX, trainY, testX, testY


# In[ ]:


# scale pixels
def prep_pixels(train, test):
    # convert from integers to floats
    train_norm = train.astype('float32')
    test_norm = test.astype('float32')
    # normalize to range 0-1
    train_norm = train_norm / 255.0
    test_norm = test_norm / 255.0
    # return normalized images
    return train_norm, test_norm


# In[ ]:


# Import ConfigSpace and different types of parameters
from smac.configspace import ConfigurationSpace
from ConfigSpace.hyperparameters import CategoricalHyperparameter,     UniformFloatHyperparameter, UniformIntegerHyperparameter
from ConfigSpace.conditions import InCondition

# Import SMAC-utilities
from smac.tae.execute_func import ExecuteTAFuncDict
from smac.scenario.scenario import Scenario
from smac.facade.smac_facade import SMAC


# We load the iris-dataset (a widely used benchmark)
iris = datasets.load_iris()

def svm_from_cfg(cfg):
    cfg = {k : cfg[k] for k in cfg if cfg[k]}
    clf = svm.SVC(kernel='rbf', **cfg, random_state=42)
    scores = cross_val_score(clf, iris.data, iris.target, cv=5)
    return 1-np.mean(scores)  # Minimize!


# In[ ]:


def get_training_data():
    train_data,dev_data,test_data = read_files()
    train_data,train_label,_,_,_,_ = preprocessing_sklearn(train_data,dev_data,test_data)
    return train_data,train_label

def cross_validation(depth,size):
    train_data,train_label = get_training_data()
    mean_validation_score = cross_val_score(
        BaggingClassifier(DecisionTreeClassifier(max_depth = int(depth),random_state=1),n_estimators = int(size)).fit(train_data, train_label),
        train_data,train_label, cv=2).mean()
    return mean_validation_score

def bagging(train_data,train_label,test_data,test_label,depth_of_tree,size_of_ensemble):
    decision_tree = DecisionTreeClassifier(max_depth = int(round(depth_of_tree)),random_state=1)
    model = BaggingClassifier(decision_tree,n_estimators = int(round(size_of_ensemble))).fit(train_data, train_label)
    predicted = model.predict(test_data)
    accuracy = accuracy_score(test_label,predicted)*100
    return accuracy

def bayes_optimisation(number_of_iterations):
    parameters_to_search = parameters()
    bayesopt = BayesianOptimization(cross_validation,parameters_to_search)
    bayesopt.maximize(n_iter=number_of_iterations)
    best_parameters = bayesopt.res['max']['max_params']
    return best_parameters

def accuracies(train_data,train_label,test_data,test_label):
    all_hyper_parameters = []
    accuracies = []
    iteration_range = range(10)
    iterations = []
    for iteration_number in iteration_range:
        hyper_parameters = bayes_optimisation(iteration_number+1)
        accuracy = bagging(train_data,train_label,test_data,test_label,hyper_parameters['depth'],hyper_parameters['size'])
        accuracies.append(accuracy)
        all_hyper_parameters.append(hyper_parameters)
        iterations.append(iteration_number+1)
    return accuracies,all_hyper_parameters,iterations

def print_accuracies(accuracies,hyper_parameters):
    print("Accuracies: Bagging", file =f)
    accuracies = pd.DataFrame(accuracies)
    hyp = pd.DataFrame(hyper_parameters).T
    depth_of_tree = hyp.iloc[0]
    size_of_ensemble = hyp.iloc[1]
    depth_of_tree = pd.DataFrame(depth_of_tree)
    size_of_ensemble = pd.DataFrame(size_of_ensemble)
    concat_data = pd.concat([depth_of_tree,size_of_ensemble,accuracies],axis=1)
    df = pd.DataFrame(concat_data)
    df.columns = ["Depth of Tree","Ensemble Size","Validation Accuracy"]
    print(df, file =f)

def plot_curve(accuracies,hyper_parameters,iterations):
    plt.plot(iterations, accuracies, color = 'red', marker='o', linestyle='solid')
    plt.xlabel("Number of Iterations")
    plt.ylabel("Accuracy (%)")
    plt.title("Validation Accuracy vs Number of Iterations: Bagging") 

    for label, x, y in zip(hyper_parameters, iterations, accuracies):
        plt.text(x,y,label,fontsize=8,horizontalalignment='center',verticalalignment='center')

    save_file_name = "Accuracy_vs_Number_of_Iterations_Bagging.png" 
    plt.savefig(save_file_name)
    #plt.show()


# In[ ]:


cs = ConfigurationSpace()
C = UniformFloatHyperparameter("C", 0.001, 1000.0, default_value=1.0)
cs.add_hyperparameter(C)
scenario = Scenario({"run_obj": "quality",   # we optimize quality (alternatively runtime)
                     "runcount-limit": 50,  # maximum function evaluations
                     "cs": cs,               # configuration space
                     "deterministic": "true"
                     })


# In[ ]:


smac = SMAC(scenario=scenario, rng=np.random.RandomState(42),
        tae_runner=svm_from_cfg)
incumbent = smac.optimize()
print(incumbent)
inc_value = svm_from_cfg(incumbent)


# In[ ]:


print("Optimized Value: %.2f" % (inc_value))


# In[ ]:


def main():
    train_data,dev_data,test_data = read_files()
    train_data_sk,train_label_sk,dev_data_sk,dev_label_sk,_,_ = preprocessing_sklearn(train_data,dev_data,test_data)
    accuracy,hyper_parameters,iterations = accuracies(train_data_sk,train_label_sk,dev_data_sk,dev_label_sk)
    
    print("Bagging Bayesian Optimisation:", file =f)
    print(file =f)
    print_accuracies(accuracy,hyper_parameters)
    plot_curve(accuracy,hyper_parameters,iterations)

main()

f.close()

