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
# Importing the csv file
input_file = "symptoms_all.csv"
cols = [0, 8, 22, 25, 37, 45, 47, 64, 101, 116, 253, 280]

df = pd.read_csv(input_file, header=None, keep_default_na=False, dtype=str)

if df.shape[1] <= max(cols):
    st.error(f"Need at least {max(cols)+1} columns, but file has {df.shape[1]}.")
else:
    dataset = df.iloc[:, cols].copy()
    dataset = dataset.replace({"": "0", "y": "1"})
    dataset.columns = [
        "Id", "female", "tuber", "diabetes", "men_con", "cough",
        "ch_cough", "diarr", "exc_urine", "exc_drink",
        "disease_description", "finaldiagnosis"
    ]

    dataset["finaldiagnosis"] = pd.to_numeric(
        dataset["finaldiagnosis"], errors="coerce"
    ).fillna(0).astype(int)

    st.write(dataset)

    dataset1 = dataset[["disease_description", "finaldiagnosis"]]
    dataset2 = dataset[["female", "tuber", "diabetes", "men_con", "cough",
                        "ch_cough", "diarr", "exc_urine", "exc_drink"]]
