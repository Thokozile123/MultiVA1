# Importing necessary libraries
import re
import logging
import multiprocessing
import urllib.request
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
import matplotlib.pyplot as plt
import nltk
import gensim
import bs4 as bs
import cleantext as clean

from patsy import dmatrices
import statsmodels.api as sm

from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk.tokenize import sent_tokenize, word_tokenize

from gensim.models import KeyedVectors, Doc2Vec
from gensim.models.doc2vec import TaggedDocument

from sklearn import metrics, datasets, svm, utils
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, roc_curve, fbeta_score, make_scorer, recall_score, precision_score, f1_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier

from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTETomek
from imblearn.pipeline import make_pipeline

#from hyperas import optim
#from hyperas.distributions import choice, uniform
from hyperopt import Trials, STATUS_OK, tpe
from tqdm import tqdm
st.write("Hi Thokozile")

# Importing the csv file
import csv
input_file= "symptoms_all.csv"
output_file="symptoms_all_variables.csv"
with open(input_file, 'r') as csvfile:
    with open(output_file, 'w', newline='') as f:
        spamreader = csv.reader(csvfile, delimiter=',')
        for row in spamreader:
            need = [row[0], row[8],row[22],row[25],row[37],row[45],row[47],row[64],row[101], row[116], row[253], row[280]]
            len_need = len(need)
            for i in range(len_need):
                if need[i] == '':
                   need[i] = "0"
                elif need[i] == 'y':
                   need[i] = "1"
                else:
                   continue

            if need[0] == 'y':
               need[0] = "1"
            #print(need)
dataset = pd.read_csv(output_file)
dataset
            thewriter = csv.writer(f      thewriter.writerow(need)
