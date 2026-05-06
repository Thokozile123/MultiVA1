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


import csv
import pandas as pd

input_file = "symptoms_all.csv"
output_file = "symptoms_all_variables.csv"
cols = [0, 8, 22, 25, 37, 45, 47, 64, 101, 116, 253, 280]

with open(input_file, "r", newline="", encoding="utf-8") as csvfile, \
     open(output_file, "w", newline="", encoding="utf-8") as f:
    reader = csv.reader(csvfile)
    writer = csv.writer(f)

    for row in reader:
        if len(row) <= max(cols):
            continue
        need = [row[i] for i in cols]
        need = ["0" if x == "" else "1" if x == "y" else x for x in need]
        writer.writerow(need)

#Reading the file
dataset = pd.read_csv(output_file, header=None)
dataset

dataset.columns = dataset.columns.str.strip()
print(dataset.columns.tolist())

#Changing the target into integer
#dataset["finaldiagnosis"] = dataset["finaldiagnosis"].astype(int)
#dataset





