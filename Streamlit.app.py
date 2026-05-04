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
#import texthero as hero

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
