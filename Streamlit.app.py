# ================================
# IMPORT LIBRARIES
# ================================
import multiprocessing
import numpy as np
import pandas as pd
import streamlit as st
import nltk

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize

from gensim.models import Doc2Vec
from gensim.models.doc2vec import TaggedDocument

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    KFold
)

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    recall_score,
    precision_score,
    f1_score
)

from sklearn.ensemble import RandomForestClassifier

from imblearn.combine import SMOTETomek
from imblearn.over_sampling import SMOTE

# ================================
# NLTK DOWNLOADS
# ================================
nltk.download("stopwords")
nltk.download("punkt")

# ================================
# STREAMLIT TITLE
# ================================
st.title("MULTIVA Classification App")
st.write("Hi Thokozile")

# ================================
# FILE UPLOAD
# ================================
uploaded_file = st.file_uploader(
    "Upload symptoms_all.csv",
    type=["csv"]
)

if uploaded_file is not None:

    # ================================
    # LOAD DATA
    # ================================
    try:
        df = pd.read_csv(
            uploaded_file,
            header=None,
            keep_default_na=False,
            dtype=str
        )

    except pd.errors.EmptyDataError:
        st.error("The uploaded CSV file is empty.")
        st.stop()

    except Exception as e:
        st.error(f"Error reading CSV file: {e}")
        st.stop()

    # ================================
    # SELECT REQUIRED COLUMNS
    # ================================
    cols = [0, 8, 22, 25, 37, 45, 47, 64, 101, 116, 253, 280]

    if df.shape[1] <= max(cols):
        st.error(
            f"Need at least {max(cols)+1} columns, "
            f"but file has {df.shape[1]} columns."
        )
        st.stop()

    dataset = df.iloc[:, cols].copy()

    # ================================
    # CLEAN RAW VALUES
    # ================================
    dataset = dataset.replace({
        "": "0",
        "y": "1",
        "Y": "1"
    })

    dataset.columns = [
        "Id",
        "female",
        "tuber",
        "diabetes",
        "men_con",
        "cough",
        "ch_cough",
        "diarr",
        "exc_urine",
        "exc_drink",
        "disease_description",
        "finaldiagnosis"
    ]

    # ================================
    # TARGET VARIABLE
    # ================================
    dataset["finaldiagnosis"] = pd.to_numeric(
        dataset["finaldiagnosis"],
        errors="coerce"
    ).fillna(0).astype(int)

    st.header("Dataset")
    st.write(dataset.head())

    # ================================
    # SPLIT FEATURES
    # ================================
    dataset1 = dataset[
        ["disease_description", "finaldiagnosis"]
    ]

    dataset2 = dataset[
        [
            "female",
            "tuber",
            "diabetes",
            "men_con",
            "cough",
            "ch_cough",
            "diarr",
            "exc_urine",
            "exc_drink"
        ]
    ]

    st.header("Narrative Features")
    st.write(dataset1.head())

    st.header("Binary Features")
    st.write(dataset2.head())

    # ================================
    # CLEAN BINARY FEATURES
    # ================================
    dataset21 = dataset2.replace("#NULL!", np.nan)

    dataset22 = dataset21.fillna(-1)

    # Convert ALL columns to numeric
    dataset22 = dataset22.apply(
        pd.to_numeric,
        errors="coerce"
    )

    dataset22 = dataset22.fillna(-1)

    st.header("Binary Features After Cleaning")
    st.write(dataset22.head())

    st.write("Data Types:")
    st.write(dataset22.dtypes)

    # ================================
    # TEXT CLEANING
    # ================================
    text = dataset1["disease_description"].fillna("").astype(str)

    dataset11 = text.str.lower()

    dataset11 = dataset11.str.replace(
        r"[^a-z0-9\s]",
        " ",
        regex=True
    )

    dataset11 = dataset11.str.replace(
        r"\s+",
        " ",
        regex=True
    ).str.strip()

    dataset12 = dataset11.str.replace(
        r"\d+",
        "",
        regex=True
    )

    # ================================
    # STOPWORDS
    # ================================
    default_stopwords = set(stopwords.words("english"))

    custom_stopwords = default_stopwords.union({
        "sugar", "suger", "suggar", "sugr",
        "sugra", "shugar", "sujer", "sogur",
        "suagr", "sruag", "sguar", "suar",
        "suga", "sgar", "diabetes",
        "diabetis", "diabetees", "daiabetes",
        "diabtees", "diabets", "dieabetes",
        "deabetes", "diabtes", "dyabetes",
        "diabate", "diabetic", "diabete",
        "dabetes", "diabees"
    })

    def remove_stopwords_from_text(s):
        tokens = s.split()
        tokens = [
            t for t in tokens
            if t not in custom_stopwords
        ]
        return " ".join(tokens)

    data3 = dataset12.apply(remove_stopwords_from_text)

    # ================================
    # FINAL TEXT DATAFRAME
    # ================================
    datatrain = pd.concat(
        [
            data3.rename("disease_description"),
            dataset1["finaldiagnosis"]
        ],
        axis=1
    )

    st.header("Narrative Features After Cleaning")
    st.write(datatrain.head())

    # ================================
    # TOKENIZATION
    # ================================
    def tokenize_text(text):
        tokens = []

        for sent in sent_tokenize(text):
            for word in word_tokenize(sent):

                if len(word) < 2:
                    continue

                tokens.append(word.lower())

        return tokens

    # ================================
    # TAG DOCUMENTS
    # ================================
    datatraintagged = datatrain.apply(
        lambda r: TaggedDocument(
            words=tokenize_text(r["disease_description"]),
            tags=[r.finaldiagnosis]
        ),
        axis=1
    )

    # ================================
    # DOC2VEC TRAINING
    # ================================
    cores = 1

    @st.cache_resource
    def train_doc2vec_models(tagged_docs):

        # DBOW
        model_dbow = Doc2Vec(
            tagged_docs,
            dm=0,
            vector_size=50,
            negative=5,
            hs=0,
            min_count=2,
            workers=1,
            epochs=30
        )

        # DMM
        model_dmm = Doc2Vec(
            tagged_docs,
            dm=1,
            dm_mean=1,
            vector_size=50,
            window=10,
            workers=1,
            epochs=30
        )

        return model_dbow, model_dmm

    model_dbow, model_dmm = train_doc2vec_models(
        datatraintagged
    )

    st.write(
        "DBOW vocab size:",
        len(model_dbow.wv)
    )

    st.write(
        "DMM vocab size:",
        len(model_dmm.wv)
    )

    # ================================
    # CONCATENATED DOC2VEC
    # ================================
    class ConcatenatedDoc2Vec:

        def __init__(self, dbow_model, dmm_model):
            self.dbow = dbow_model
            self.dmm = dmm_model
            self.vector_size = (
                self.dbow.vector_size * 2
            )

        def infer_vector(
            self,
            doc_words,
            alpha=0.025,
            steps=100
        ):

            dbow_vec = self.dbow.infer_vector(
                doc_words,
                alpha=alpha,
                epochs=steps
            )

            dmm_vec = self.dmm.infer_vector(
                doc_words,
                alpha=alpha,
                epochs=steps
            )

            return np.hstack(
                (dbow_vec, dmm_vec)
            )

    combined_model = ConcatenatedDoc2Vec(
        model_dbow,
        model_dmm
    )

    # ================================
    # VECTOR CREATION
    # ================================
    def vec_for_learning(model, tagged_docs):

        targets = []
        regressors = []

        for doc in tagged_docs:

            targets.append(doc.tags[0])

            regressors.append(
                model.infer_vector(
                    doc.words,
                    steps=20
                )
            )

        return targets, regressors

    # ================================
    # TEXT FEATURES
    # ================================
    y_text, x_text = vec_for_learning(
        combined_model,
        datatraintagged
    )

    x1 = pd.DataFrame(x_text)

    # ================================
    # BINARY FEATURES
    # ================================
    x2 = dataset22

    y1 = datatrain["finaldiagnosis"]

    # ================================
    # COMBINED FEATURES
    # ================================
    dataset3 = pd.concat(
        [x1, x2],
        axis=1
    )

    # ================================
    # TRAIN TEST SPLIT
    # ================================
    x_train, x_test, y_train, y_test = train_test_split(
        x2,
        y1,
        test_size=0.2,
        random_state=42,
        stratify=y1
    )

    # ================================
    # RANDOM FOREST
    # ================================
    rf_params = {
        "n_estimators": 100,
        "max_depth": 5,
        "random_state": 13
    }

    rf_model = RandomForestClassifier(
        **rf_params
    )

    # ================================
    # CROSS VALIDATION
    # ================================
    kf = StratifiedKFold(
        n_splits=5,
        random_state=42,
        shuffle=True
    )

    # ================================
    # SMOTE + TRAINING
    # ================================
    smoter = SMOTETomek(
        smote=SMOTE(k_neighbors=1),
        random_state=42
    )

    recalls = []
    precisions = []
    f1s = []
    aucs = []
    accuracies = []

    for train_index, val_index in kf.split(
        x_train,
        y_train
    ):

        x_train_fold = x_train.iloc[train_index]
        x_val_fold = x_train.iloc[val_index]

        y_train_fold = y_train.iloc[train_index]
        y_val_fold = y_train.iloc[val_index]

        # RESAMPLE
        x_resampled, y_resampled = smoter.fit_resample(
            x_train_fold,
            y_train_fold
        )

        # TRAIN
        rf_model.fit(
            x_resampled,
            y_resampled
        )

        # PREDICT
        predictions = rf_model.predict(x_val_fold)

        # METRICS
        recall = recall_score(
            y_val_fold,
            predictions,
            zero_division=0
        )

        precision = precision_score(
            y_val_fold,
            predictions,
            zero_division=0
        )

        f1 = f1_score(
            y_val_fold,
            predictions,
            zero_division=0
        )

        accuracy = accuracy_score(
            y_val_fold,
            predictions
        )

        try:
            auc = roc_auc_score(
                y_val_fold,
                predictions
            )
        except:
            auc = np.nan

        recalls.append(recall)
        precisions.append(precision)
        f1s.append(f1)
        accuracies.append(accuracy)
        aucs.append(auc)

    # ================================
    # RESULTS
    # ================================
    st.header("Random Forest Results")

    results = pd.DataFrame({
        "Metric": [
            "Recall",
            "Precision",
            "F1 Score",
            "ROC AUC",
            "Accuracy"
        ],
        "Score": [
            np.mean(recalls),
            np.mean(precisions),
            np.mean(f1s),
            np.nanmean(aucs),
            np.mean(accuracies)
        ]
    })

    st.write(results)

    # ================================
    # FINAL MODEL FIT
    # ================================
    rf_model.fit(x_train, y_train)

    final_predictions = rf_model.predict(x_test)

    st.header("Final Test Predictions")

    prediction_df = pd.DataFrame({
        "Actual": y_test.values,
        "Predicted": final_predictions
    })

    st.write(prediction_df.head(20))

    st.success("Model training completed successfully.")
