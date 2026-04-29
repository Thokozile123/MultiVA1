# Importing necessary libraries

import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn import metrics
from patsy import dmatrices
import statsmodels.api as sm
from keras import models
from keras import layers
from keras.layers import Dense
from keras.models import Model
import matplotlib.pyplot as plt
from keras.layers import Dropout
from keras.layers import Flatten
from keras.optimizers import SGD
from keras.layers import Flatten
from nltk.corpus import stopwords
from sklearn import datasets, svm
from keras.models import Sequential
from keras.layers import Embedding
from pandas import Series, DataFrame
from keras.layers import MaxPooling1D
from gensim.models import KeyedVectors
from keras.utils import to_categorical
from sklearn.linear_model import SGDClassifier
#from keras.preprocessing.text import Tokenizer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from statsmodels.nonparametric import smoothers_lowess
from keras.preprocessing.sequence import pad_sequences
from statsmodels.nonparametric.kde import KDEUnivariate
from keras.wrappers.scikit_learn import KerasClassifier
from keras.layers import Conv1D, GlobalMaxPooling1D, Conv2D, MaxPooling2D
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, roc_curve
from sklearn.naive_bayes import MultinomialNB

import re
import nltk
import gensim
import logging
import bs4 as bs
import urllib.request
import multiprocessing
import texthero as hero
from gensim import utils
from sklearn import utils
from hyperas import optim
from tqdm import tqdm
tqdm.pandas(desc="progress-bar")
from gensim.models import Doc2Vec
from nltk.corpus import stopwords
from texthero import stopwords
from imblearn import datasets
from nltk.stem import SnowballStemmer
from sklearn.metrics import accuracy_score
from hyperopt import Trials, STATUS_OK, tpe
from gensim.models.doc2vec import TaggedDocument
from hyperas.distributions import choice, uniform
from gensim.models.doc2vec import LabeledSentence
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTETomek
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import fbeta_score, make_scorer
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import sent_tokenize, word_tokenize
from imblearn.pipeline import make_pipeline
from sklearn.metrics import recall_score, roc_auc_score, precision_score, f1_score
from sklearn.model_selection import cross_val_score, GridSearchCV, train_test_split, StratifiedKFold

st.title('MultiVA')
st.info('This is a verbal autopsy text classification app')
