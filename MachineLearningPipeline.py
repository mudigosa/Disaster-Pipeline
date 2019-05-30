#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
import pandas as pd
import numpy as np
import nltk
import os

from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support

from sqlalchemy import create_engine

import pickle
import matplotlib.pyplot as plt
from sklearn import multioutput
nltk.download('punkt')
nltk.download('wordnet')


# In[5]:


def load_data():
        engine = create_engine('sqlite:///disastermessages.db')
        X, Y, category_names = load_data(engine)
        print(X)
        print(Y)
        return X,Y   


# In[2]:


engine = create_engine('sqlite:///disastermessages.db')
df = pd.read_sql_table('disastermessage', engine)
df.head()
X = df.loc[:, 'message'].astype(str)
print(X)
Y = df.iloc[:, 4:].astype(str)
categories = list(Y)
print(categories)
display (Y.head (n=3))


# In[3]:


engine = create_engine('sqlite:///disastermessages.db')
df = pd.read_sql_table('disastermessage', engine)
df.head()
X = df.loc[:, 'message'].astype(str)
print(X)
Y = df.iloc[:, 4:].astype(str)
categories = list(Y)
print(categories)
display (Y.head (n=3))


# In[73]:


fig = plt.figure(figsize = (15,20))
ax = fig.gca()
Y.hist(ax = ax)


# In[25]:


df.head()


# In[4]:


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
X_train = X_train.astype(str)
Y_train = Y_train.astype(str)
print(X_train)


# In[7]:


pipeline = Pipeline([
            ('vect', CountVectorizer(tokenizer=tokenize)),
            ('tfidf', TfidfTransformer()),
    #        ('clf', multioutput.MultiOutputClassifier (RandomForestClassifier(), n_jobs = 35))
            ('clf', multioutput.MultiOutputClassifier (RandomForestClassifier()))
            ])


# In[10]:


import nltk
model = build_pipeline_model()
print('good')
pipeline.fit(X_train, Y_train)        
    



# In[26]:


y_pred_test = pipeline.predict(X_test)
y_pred_train = pipeline.predict(X_train)
for i in range(len(category_names)):
        print("Category:", category[i],"\n", classification_report(Y_test.iloc[:, i].values, Y_pred[:, i]))
        print('Accuracy of %25s: %.2f' %(category[i], accuracy_score(Y_test.iloc[:, i].values, Y_pred[:,i])))


# In[6]:


def tokenize(text):
    '''
    param: text
    return: list of tokens
    '''

    tokens = word_tokenize(text)
    wl = WordNetLemmatizer()
    # converting to lower case and lemmatize each token
    tokens = [wl.lemmatize(t).lower().strip() for t in tokens]

    return tokens


# In[9]:


def build_pipeline_model():
    '''
    Builds the model pipeline
    param:parameters for RF model
    returns:pipeline
    '''

   #setting pipeline
    pipeline = Pipeline([
            ('vect', CountVectorizer(tokenizer=tokenize)),
            ('tfidf', TfidfTransformer()),
    #        ('clf', multioutput.MultiOutputClassifier (RandomForestClassifier(), n_jobs = 35))
            ('clf', multioutput.MultiOutputClassifier (RandomForestClassifier()))
            ])

    X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state = 25)
    
    return pipeline


# In[20]:


def evaluate_pipeline_model(model, X_test, Y_test):
    """
    Evaluates model's performance on the test set.
    param : model
    param : X_test
    param : Y_test
    param : category_names
    return : None
    """
    predictions = model.predict(X_test)
    
    # Since there are 36 categories, we'll just loop over them to calculate the accuracy of each category.
    print("Accuracy scores for each category\n")
    print("*-" * 30)

    for i in range(36):
        print("Category:", category_names[i],"\n", classification_report(Y_test[:, i], predictions[:, i]))


# In[16]:


def main():
        engine = create_engine('sqlite:///disastermessages.db')
        df = pd.read_sql_table('disastermessage', engine)
        X = df['message']
        Y = df.iloc[:,4:]
        print('Loading data...\n    DATABASE: {}'.format(engine))
        X, Y, category_names = load_data(engine)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

        print('Building random forest model with best parameters')
        model = build_pipeline_model()

        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

