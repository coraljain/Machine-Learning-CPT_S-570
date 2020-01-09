#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np 
import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# In[ ]:


from functools import reduce
import operator
f = open('program_output.txt', 'r')


# In[ ]:


def naive_bayes(probability_dictionary,feature_matrix,probability_wise_total,probability_future_total,type_of_data):
    class_labels = feature_matrix.iloc[:,-1]
    feature_matrix = feature_matrix.drop('target',1)
    number_of_rows = feature_matrix.shape[0]
    words = probability_dictionary.keys()
    number_of_correct_predictions = 0
    number_of_incorrect_predictions = 0
    for index in range(number_of_rows):
        probability_wise = 1
        probability_future = 1
        data_to_traverse = feature_matrix.iloc[index]
        for word in words:
            if data_to_traverse[word] == 1:
                probability_wise*=probability_dictionary[word]['present']['wise']
                probability_future*=probability_dictionary[word]['present']['future']

        wise_prob = probability_wise*probability_wise_total
        future_prob = probability_future*probability_future_total
        if wise_prob > future_prob:
            label = 0
        else:
            label = 1
        if(label == class_labels.iloc[index]):
            number_of_correct_predictions+=1
        else:
            number_of_incorrect_predictions+=1
    accuracy = probability_calculation(number_of_correct_predictions,number_of_rows)
    print("Naive Bayes %s accuracy: %f" %(type_of_data, accuracy*100),file = f)


# In[ ]:


def sklearn_nb(training_data,testing_data,type_of_data):
    
    training_class_label = training_data.iloc[:,-1]
    testing_class_label = testing_data.iloc[:,-1]
    training_data = training_data.drop('target',1)
    testing_data = testing_data.drop('target',1)
    model = MultinomialNB().fit(training_data, training_class_label)
    predicted = model.predict(testing_data)
    accuracy = accuracy_score(testing_class_label,predicted)
    print("Naive Bayes(Sklearn) %s accuracy: %f" %(type_of_data, accuracy*100),file = f)


# In[ ]:


def sklearn_lr(training_data,testing_data,type_of_data):
    
    data_training = training_data.drop('target',1)
    data_testing = testing_data.drop('target',1)
    
    label_testing = testing_data.iloc[:,-1]
    label_training = training_data.iloc[:,-1]
    
    predicted = (LogisticRegression().fit(training_data, training_class_label)).predict(testing_data)
    accuracy = accuracy_score(testing_class_label,predicted)
    print("Logistic Regression(Sklearn) %s accuracy: %f" %(type_of_data, accuracy*100), file =f)


# In[ ]:


def fmatrix(dataset,vocabulary):
    data = []
    for row_data in dataset:
        word_list = dict.fromkeys(vocabulary, 0)
        for word in row_data:
            if word in word_list:
                word_list[word] = 1
        data.append(word_list)
    feature_matrix = pd.DataFrame(data)
    return feature_matrix


# In[ ]:


def convert_dict(data):
    
    set_of_words = []
    
    x = data.values.tolist()
    data_set = reduce(operator.add, x)
    
    for y in data_set:
        word = y.split()
        set_of_words.append(word)
    return set_of_words


# In[ ]:


def delete_stopwords(dataset,stopwords):
    
    data = []
    complete_data = []
    dataset = reduce(operator.add, dataset)
    stopwords = reduce(operator.add, stopwords)
    for word in dataset: 
        if word not in stopwords:
            all_data.append(word)
    for word in all_data:
        if word not in data:
            data.append(word)
    data.sort()
    return data


# In[ ]:


def null_check(value):
    if value == 0:
        return 0
    else:
        return value


# In[ ]:


def laplace_smoothing(value,total_count):
    
    v = (value+1)/(total_count+2)
    probability = null_check(v)
    return probability


# In[ ]:


def probability_calculation(value,total_count):
    val = value/total_count
    probability = null_check(val)
    return probability


# In[ ]:


def calculate_wise_future(dataframe,target,column_name,value):
    
    dataframe['target'] = target
    
    if value is not None:
        wise = len(dataframe.loc[(dataframe[column_name] == value) & (dataframe['target'] == 0)])
        future = len(dataframe.loc[(dataframe[column_name] == value) & (dataframe['target'] == 1)])
    else:
        wise = len(dataframe.loc[dataframe['target'] == 0])
        future = len(dataframe.loc[dataframe['target'] == 1])
    return wise,future,wise+future


# In[ ]:


def probability_words(feature_matrix,vocabulary,target):
    probability_dictionary = {}
    for word in vocabulary:
        dictionary = {word:{'present':{'wise':{},'future':{}},'absent':{'wise':{},'future':{}}}}
        for value in [0,1]:
            number_of_wise,number_of_future,total = calculate_wise_future(feature_matrix,target,word,value)
            probability_wise = laplace_smoothing(number_of_wise,total)
            probability_future = laplace_smoothing(number_of_future,total)
            if value == 1:
                dictionary[word]['present']['wise'] = probability_wise
                dictionary[word]['present']['future'] = probability_future
            else:
                dictionary[word]['absent']['wise'] = probability_wise
                dictionary[word]['absent']['future'] = probability_future
        probability_dictionary.update(dictionary)

    number_of_wise,number_of_future,total = calculate_wise_future(feature_matrix,target,None,None)
    probability_wise_total = probability_calculation(number_of_wise,total)
    probability_future_total = probability_calculation(number_of_future,total)

    return probability_dictionary,probability_wise_total,probability_future_total


# In[ ]:


#Fetching the data from the data available 
stop_words = pd.read_csv("stoplist.txt",header=None)
data_test = pd.read_csv("testdata.txt",header=None)
labels_test = pd.read_csv("testlabels.txt",header=None)
data_train = pd.read_csv("traindata.txt",header=None)
labels_train = pd.read_csv("trainlabels.txt",header=None)


# In[ ]:


train_dict = convert_dict(data_train)
test_dict = convert_dict(data_test)
stopwords = convert_dict(stop_words)

vocab = delete_stopwords(train_dict,stopwords)
train_fmatrix = fmatrix(train_dictionary,vocabulary)
test_fmatrix = fmatrix(test_dictionary,vocabulary)
test_fmatrix['target'] = test_label
probability_dict,probability_wise_total,probability_future_total = probability_words(feature_matrix_train,vocabulary,train_label)


print(file =f)
naive_bayes(probability_dictionary,feature_matrix_train,probability_wise_total,probability_future_total,"training")
naive_bayes(probability_dictionary,feature_matrix_test,probability_wise_total,probability_future_total,"testing")
print(file =f)
sklearn_nb(feature_matrix_train,feature_matrix_train,"training")
sklearn_nb(feature_matrix_train,feature_matrix_test,"testing")
print(file =f)
print("Logistic Regression:",file =f)
print(file =f)
sklearn_lr(feature_matrix_train,feature_matrix_train,"training")
sklearn_lr(feature_matrix_train,feature_matrix_test,"testing")

f.close()

