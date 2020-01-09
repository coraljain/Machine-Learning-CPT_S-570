#!/usr/bin/env python
# coding: utf-8

# In[177]:


#Importing Libraries

from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# In[178]:


#Calculation of Entropy 

def entropy_calculation(columns):
    
    elements, counts = np.unique(columns, return_counts=True)
    
    #expression for calculation of entropy
    entropys = np.sum([(-counts[i]/np.sum(counts))*np.log2(counts[i]/np.sum(counts)) for i in range(len(elements))])
    return entropys


# In[179]:


#Calculation of Information Gain

def Information_Gain(sub_data, feature, tar):
    
    sub = sub_data[name2]
    
    entropy = entropy_calculation(sub)

    elements, counts = np.unique(data[name1], return_counts=True)

    #weight calculation using subdata, feature and counts
    w = np.sum([(counts[i]/np.sum(counts))*
            entropy_calculation(sub.where(sub[name1] == elements[i]).dropna()[name2])
                     for i in range(len(elements))])

    IG = entropy - w
    
    return IG


# In[180]:


#Prediction of tree

def prediction(groups, tree, default = 4.0):
    
    for x in list(groups.keys()):
        if x in list(tree.keys()):
            try:
                result = tree[x][groups[key]]
            except:
                return default
            result = tree[x][groups[key]]
            if isinstance(result, dict):
                return prediction(groups, result)
            else:
                return predicted_value


# In[181]:


#Extraction of Breast Cancer Data

def dataset_extraction(location):
    
    dataset = pd.read_csv(location, names=['id', 'CT', 'UoCS', 'UoCS2', 'MA','SECS', 'BN', 'BC', 'NN', 'M', 'class'],header=None)
    
    dataset = dataset.drop('id', axis=1)
    
    training_dataset, testing_dataset = train_test_split(dataset, test_size=0.3)
    validation_dataset, testing_dataset = train_test_split(testing_dataset, test_size=0.333)

    return training_dataset, validation_dataset, testing_dataset


# In[182]:


#ID3 function program

def Decision_Tree(sub, data, features, target="class", node=None):
    
    if len(data) == 0:
        
        tree1 = np.unique(data[target])[
            np.argmax(np.unique(data[target], return_counts=True)[1])]
        
        return tree1 
    
    if len(np.unique(data[target])) <= 1:
        
        tree2 = np.unique(data[target])[0]
        
        return tree2

    if len(features) == 0:
        
        tree3 = node
        
        return tree3

    else:
        node = np.unique(sub[target])[np.argmax(np.unique(
            sub[target], return_counts=True)[1])]

        items = [IG(sub, feature, target) for feature in features]
        F = features[np.argmax(items)]

        tree = {F: {}}

        features = [i for i in features if i != F]

        for value in np.unique(sub[F]):
            sub_data = sub.where(sub[F] == value).dropna()
            subtree = Decision_Tree(sub, data, features, target, node)

            tree[F][value] = subtree

        return tree


# In[183]:


#Calculation of accuracy

def accuracy_calcuation(data, data_name, decision_tree):
    
    groups = data.iloc[:, :-1].to_dict(orient="records")
    prediction = np.zeros(len(data))
    for i in range(len(data)):
        prediction[i] = prediction(groups[i], decision_tree, 4.0)
    print(data_name,(np.sum(prediction == data["class"])/len(data)))


# In[184]:


#Main function for extracting data and calling ID3 Decision Tree function

if __name__ == "__main__":
    
    decision_tree_result = Decision_Tree(data_train, data_train, data_train.columns[:-1])
    
    print(decision_tree_result)
    
    train_data, val_data, test_data = dataset_extraction('/Users/coraljain/Downloads/breast-cancer-wisconsin.data')
    
    accuracy_calcuation(train_data, 'The Accuracy of Training Data:', decision_tree_result)
    accuracy_calcuation(val_data, 'The Accuracy of Validation Data:', decision_tree_result)
    accuracy_calcuation(test_data, 'The Accuracy of Testing Data:', decision_tree_result)


# In[ ]:




