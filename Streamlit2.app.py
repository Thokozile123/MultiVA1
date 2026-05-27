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
nltk.download("stopwords")
nltk.download("punkt_tab")
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
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedKFold, KFold
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, roc_curve, fbeta_score, make_scorer, recall_score, precision_score, f1_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier

from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTETomek
from imblearn.pipeline import make_pipeline

from hyperopt import Trials, STATUS_OK, tpe
from tqdm import tqdm

st.write("Hi Thokozile")

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

    st.header("Dataset")
    st.write(dataset)

    dataset1 = dataset[["disease_description", "finaldiagnosis"]]
    dataset2 = dataset[["female", "tuber", "diabetes", "men_con", "cough",
                        "ch_cough", "diarr", "exc_urine", "exc_drink"]]

    st.header("Narrative Features")
    st.write(dataset1)
    st.header("Binary Features")
    st.write(dataset2)

    dataset21 = dataset2.replace("#NULL!", np.nan)
    dataset22 = dataset21.fillna(-1)

    text = dataset1["disease_description"].fillna("").astype(str)

    dataset11 = text.str.lower()
    dataset11 = dataset11.str.replace(r"[^a-z0-9\s]", " ", regex=True)
    dataset11 = dataset11.str.replace(r"\s+", " ", regex=True).str.strip()
    dataset12 = dataset11.str.replace(r"\d+", "", regex=True)

    default_stopwords = set(stopwords.words("english"))
    custom_stopwords = default_stopwords.union({
        "sugar", "suger", "suggar", "sugr", "sugra", "shugar", "sujer",
        "sogur", "suagr", "sruag", "sguar", "suar", "suga", "sgar",
        "diabetes", "diabetis", "diabetees", "daiabetes", "diabtees",
        "diabets", "dieabetes", "deabetes", "diabtes", "dyabetes",
        "diabate", "diabetic", "diabete", "dabetes", "diabees"
    })

    def remove_stopwords_from_text(s):
        tokens = s.split()
        tokens = [t for t in tokens if t not in custom_stopwords]
        return " ".join(tokens)

    data3 = dataset12.apply(remove_stopwords_from_text)
    data4 = data3

    data6 = dataset1[["finaldiagnosis"]]
    datatrain = pd.concat([data4.rename("disease_description"), data6], axis=1)

    st.header("Narrative Features After Text Cleaning")
    st.write(datatrain)

    def tokenize_text(text):
        tokens = []
        for sent in nltk.sent_tokenize(text):
            for word in nltk.word_tokenize(sent):
                if len(word) < 2:
                    continue
                tokens.append(word.lower())
        return tokens

    datatraintagged = datatrain.apply(
        lambda r: TaggedDocument(
            words=tokenize_text(r["disease_description"]),
            tags=[str(r.name)]
        ),
        axis=1
    )

    cores = multiprocessing.cpu_count()

    model_dbow = Doc2Vec(
        datatraintagged.tolist(),
        dm=0,
        vector_size=50,
        negative=5,
        hs=0,
        min_count=2,
        workers=cores,
        epochs=30
    )
    model_dbow.save("dbow_model.model")
    st.write("DBOW vocab size:", len(model_dbow.wv))
    st.write("DBOW sample keys:", list(model_dbow.wv.index_to_key[:10]))

    model_dmm = Doc2Vec(
        datatraintagged.tolist(),
        dm=1,
        dm_mean=1,
        vector_size=50,
        window=10,
        workers=5,
        epochs=30
    )
    model_dmm.save("dmm_model.model")
    st.write("DMM vocab size:", len(model_dmm.wv))
    st.write("DMM sample keys:", list(model_dmm.wv.index_to_key[:10]))

    class ConcatenatedDoc2Vec:
        def __init__(self, dbow_model, dmm_model):
            self.dbow = dbow_model
            self.dmm = dmm_model
            self.vector_size = self.dbow.vector_size * 2

        def infer_vector(self, doc_words, alpha=0.025, epochs=100):
            dbow_vec = self.dbow.infer_vector(doc_words, alpha=alpha, epochs=epochs)
            dmm_vec = self.dmm.infer_vector(doc_words, alpha=alpha, epochs=epochs)
            return np.hstack((dbow_vec, dmm_vec))

    combined_model = ConcatenatedDoc2Vec(model_dbow, model_dmm)

    def vec_for_learning(model, tagged_docs):
        targets, regressors = zip(*[
            (doc.tags[0], model.infer_vector(doc.words, epochs=20))
            for doc in tagged_docs
        ])
        return targets, regressors

    y, x = vec_for_learning(combined_model, datatraintagged.tolist())
    x1 = pd.DataFrame(x)

    x2 = pd.DataFrame(dataset22)
    y1 = datatrain["finaldiagnosis"]

    dataset3 = pd.concat([x1, x2], axis=1)
    dataset4 = x1.copy()

    dataset2.columns = dataset2.columns.astype(str)
    dataset3.columns = dataset3.columns.astype(str)
    dataset4.columns = dataset4.columns.astype(str)

    kf = StratifiedKFold(n_splits=5, random_state=42, shuffle=True)

    x1_train, x1_test, y1_train, y1_test = train_test_split(
        dataset2, y1, random_state=42, stratify=y1
    )
    x2_train, x2_test, y2_train, y2_test = train_test_split(
        dataset3, y1, random_state=42, stratify=y1
    )
    x3_train, x3_test, y3_train, y3_test = train_test_split(
        dataset4, y1, random_state=42, stratify=y1
    )

    rf1 = RandomForestClassifier(n_estimators=100, random_state=42)
    example_params = {"n_estimators": 100, "max_depth": 5, "random_state": 13}
    params = {
        "n_estimators": [50, 100, 200],
        "max_depth": [4, 6, 10, 12],
        "random_state": [13]
    }

    def score_model(model, params, cv=None):
        if cv is None:
            cv = KFold(n_splits=5, shuffle=True, random_state=42)

        smoter = SMOTETomek(random_state=42)
        scores1, scores2, scores3, scores4, scores5 = [], [], [], [], []

        for train_fold_index, val_fold_index in cv.split(x1_train, y1_train):
            x_train_fold = x1_train.iloc[train_fold_index]
            y_train_fold = y1_train.iloc[train_fold_index]
            x_val_fold = x1_train.iloc[val_fold_index]
            y_val_fold = y1_train.iloc[val_fold_index]

            x_train_fold_upsample, y_train_fold_upsample = smoter.fit_resample(
                x_train_fold, y_train_fold
            )

            model_obj = model(**params).fit(x_train_fold_upsample, y_train_fold_upsample)

            y_pred = model_obj.predict(x_val_fold)
            if hasattr(model_obj, "predict_proba"):
                y_prob = model_obj.predict_proba(x_val_fold)[:, 1]
                score4 = roc_auc_score(y_val_fold, y_prob)
            else:
                score4 = roc_auc_score(y_val_fold, y_pred)

            score1 = recall_score(y_val_fold, y_pred)
            score2 = precision_score(y_val_fold, y_pred, zero_division=0)
            score3 = f1_score(y_val_fold, y_pred, zero_division=0)
            score5 = accuracy_score(y_val_fold, y_pred)

            scores1.append(score1)
            scores2.append(score2)
            scores3.append(score3)
            scores4.append(score4)
            scores5.append(score5)

        dfrf11 = pd.Series(scores1).mean()
        dfrf12 = pd.Series(scores2).mean()
        dfrf13 = pd.Series(scores3).mean()
        dfrf14 = pd.Series(scores4).mean()
        dfrf15 = pd.Series(scores5).mean()

        return dfrf11, dfrf12, dfrf13, dfrf14, dfrf15

    score_model(RandomForestClassifier, example_params, cv=kf)
