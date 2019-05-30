#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
import pandas as pd
from sqlalchemy import create_engine
import nltk
import re
import pickle
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
from sklearn import multioutput
nltk.download('punkt')
nltk.download('wordnet')

def load_data():
        engine = create_engine('sqlite:///disastermessages.db')
        df = pd.read_sql_table('disastermessage', engine)
        df.head()
        X = df.loc[:, 'message'].astype(str)
        print(X)
        Y = df.iloc[:, 4:].astype(str)
        categories = list(Y)
        print(categories)
        display (Y.head (n=3))
        return X,Y,categories 

def tokenize(text):
    '''
    Tokenize and clean text
    Input:
        text: original message text
    Output:
        lemmed: Tokenized, cleaned, and lemmatized text
    '''
    # Normalize Text
    tokens = word_tokenize(text)
    wl = WordNetLemmatizer()
    # converting to lower case and lemmatize each token
    tokens = [wl.lemmatize(t).lower().strip() for t in tokens]
    return tokens

    
def build_model(X_train, Y_train,X_test,categories):
    '''
    Build a ML pipeline using ifidf, random forest
    Input: None
    '''
    pipeline = Pipeline([
            ('vect', CountVectorizer(tokenizer=tokenize)),
            ('tfidf', TfidfTransformer()),
    #        ('clf', multioutput.MultiOutputClassifier (RandomForestClassifier(), n_jobs = 35))
            ('clf', multioutput.MultiOutputClassifier (RandomForestClassifier()))
            ])
    pipeline.fit(X_train, Y_train)
    Y_pred = pipeline.predict(X_test)
    for i in range(len(categories)):
        print("Category:", categories[i],"\n", classification_report(Y_test.iloc[:, i].values, Y_pred[:, i]))
        print('Accuracy of %25s: %.2f' %(categories[i], accuracy_score(Y_test.iloc[:, i].values, Y_pred[:,i])))
    
    # Calculate the accuracy for each of them.

def main():
        X, Y,categories = load_data()
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        X_train = X_train.astype(str)
        Y_train = Y_train.astype(str)
        print('Building random forest model with best parameters')
        model = build_model(X_train,Y_train,X_test,categories)

        print('Training model...')



if __name__ == '__main__':
    main()
