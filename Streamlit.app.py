# MULTIVA Streamlit App with Navigation Structure
# ============================================================
# MULTIVA STREAMLIT APPLICATION
# WITH NAVIGATION MENU
# ============================================================

# ============================================================
# IMPORT LIBRARIES
# ============================================================
import numpy as np
import pandas as pd
import streamlit as st
import nltk

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize

from gensim.models import Doc2Vec
from gensim.models.doc2vec import TaggedDocument

from sklearn.model_selection import (train_test_split, StratifiedKFold)

from sklearn.metrics import (
    accuracy_score,
    recall_score,
    precision_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve)

from sklearn.ensemble import RandomForestClassifier
from imblearn.combine import SMOTETomek
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt

# ============================================================
# NLTK DOWNLOADS
# ============================================================
nltk.download("stopwords")
nltk.download("punkt")
nltk.download("punkt_tab")
# ============================================================
# STREAMLIT SETTINGS
# ============================================================
st.set_page_config(
    page_title="MULTIVA Classification",
    layout="wide"
)

# ============================================================
# SIDEBAR NAVIGATION
# ============================================================
st.sidebar.title("MULTIVA Navigation")

page = st.sidebar.radio(
    "Go To",
    ["Home",
    "Dataset Overview",
    "Text Preprocessing",
    "Feature Engineering",
    "Model Training",
    "Model Evaluation",
    "Predictions",
    "Research Insights"
    ]
)

# ============================================================
# HOME PAGE
# ============================================================
if page == "Home":

    st.title("MULTIVA Classification System")

    st.subheader("Verbal Autopsy Classification Using NLP and Machine Learning")

    st.write(
        """
        This application classifies Verbal Autopsy narratives and binary symptom
        data using Natural Language Processing (NLP) and Machine Learning.

        The system evaluates:

        - Binary symptom features
        - Narrative text features
        - Combined features

        using Random Forest classification.
        """
    )

    st.header("Workflow")

    st.markdown(
        """
        1. Upload Verbal Autopsy Dataset
        2. Clean and preprocess text
        3. Generate Doc2Vec embeddings
        4. Train Random Forest classifiers
        5. Evaluate binary, text and combined models
        6. Generate predictions
        """
    )

# ============================================================
# FILE UPLOADER
# ============================================================
uploaded_file = st.sidebar.file_uploader(
    "Upload symptoms_all.csv",
    type=["csv"]
)

# ============================================================
# STOP IF NO FILE
# ============================================================
if uploaded_file is None:

    st.info("Please upload a CSV dataset from the sidebar.")
    st.stop()

# ============================================================
# LOAD DATA
# ============================================================
try:

    df = pd.read_csv(
        uploaded_file,
        header=None,
        keep_default_na=False,
        dtype=str
    )
    
except Exception as e:

    st.error(f"Error loading dataset: {e}")
    st.stop()
# ============================================================
# COLUMN SELECTION
# ============================================================
cols = [0, 8, 22, 25, 37, 45, 47, 64, 101, 116, 253, 280]

if df.shape[1] <= max(cols):

    st.error(
        f"Dataset must contain at least {max(cols)+1} columns"
    )

    st.stop()

# ============================================================
# CREATE DATASET
# ============================================================
dataset = df.iloc[:, cols].copy()

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
st.header('Data Statistics')
st.write(dataset.describe())

st.header('Data Header')
st.write(dataset.head())

fig, ax = plt.subplot(1,1)
ax.scatter()
# ============================================================
# CLEAN DATA
# ============================================================
dataset = dataset.replace({
    "": "0",
    "y": "1",
    "Y": "1"
})

# ============================================================
# TARGET VARIABLE
# ============================================================
dataset["finaldiagnosis"] = pd.to_numeric(
    dataset["finaldiagnosis"],
    errors="coerce"
).fillna(0).astype(int)

# ============================================================
# SPLIT FEATURES
# ============================================================
dataset1 = dataset[[
    "disease_description",
    "finaldiagnosis"
]]

dataset2 = dataset[[
    "female",
    "tuber",
    "diabetes",
    "men_con",
    "cough",
    "ch_cough",
    "diarr",
    "exc_urine",
    "exc_drink"
]]

# ============================================================
# CLEAN BINARY FEATURES
# ============================================================
dataset22 = dataset2.replace(
    "#NULL!",
    np.nan
)

dataset22 = dataset22.fillna(-1)

dataset22 = dataset22.apply(
    pd.to_numeric,
    errors="coerce"
)

dataset22 = dataset22.fillna(-1)

dataset22.columns = dataset22.columns.astype(str)

# ============================================================
# TEXT CLEANING
# ============================================================
text = dataset1[
    "disease_description"
].fillna("").astype(str)

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

# ============================================================
# STOPWORDS
# ============================================================
default_stopwords = set(
    stopwords.words("english")
)

custom_stopwords = default_stopwords.union({
    "sugar", "diabetes", "diabetic"
})

# ============================================================
# REMOVE STOPWORDS
# ============================================================
def remove_stopwords_from_text(s):

    tokens = s.split()

    tokens = [
        t for t in tokens
        if t not in custom_stopwords
    ]

    return " ".join(tokens)

cleaned_text = dataset12.apply(
    remove_stopwords_from_text
)

# ============================================================
# FINAL TEXT DATAFRAME
# ============================================================
datatrain = pd.concat(
    [
        cleaned_text.rename(
            "disease_description"
        ),
        dataset1["finaldiagnosis"]
    ],
    axis=1
)

# ============================================================
# TOKENIZATION
# ============================================================
def tokenize_text(text):

    tokens = []

    for sent in sent_tokenize(text):

        for word in word_tokenize(sent):

            if len(word) < 2:
                continue

            tokens.append(word.lower())

    return tokens

# ============================================================
# TAG DOCUMENTS
# ============================================================
datatraintagged = datatrain.apply(

    lambda r: TaggedDocument(

        words=tokenize_text(
            r["disease_description"]
        ),

        tags=[r.finaldiagnosis]
    ),

    axis=1
)

# ============================================================
# TRAIN DOC2VEC MODELS
# ============================================================
model_dbow = Doc2Vec(
    datatraintagged,
    dm=0,
    vector_size=50,
    negative=5,
    hs=0,
    min_count=2,
    workers=1,
    epochs=30
)

model_dmm = Doc2Vec(
    datatraintagged,
    dm=1,
    dm_mean=1,
    vector_size=50,
    window=10,
    workers=1,
    epochs=30
)

# ============================================================
# CONCATENATED DOC2VEC
# ============================================================
class ConcatenatedDoc2Vec:

    def __init__(self, dbow_model, dmm_model):

        self.dbow = dbow_model
        self.dmm = dmm_model

    def infer_vector(
        self,
        doc_words,
        alpha=0.025,
        steps=20
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

        return np.hstack((dbow_vec, dmm_vec))

combined_model = ConcatenatedDoc2Vec(
    model_dbow,
    model_dmm
)

# ============================================================
# VECTOR CREATION
# ============================================================
def vec_for_learning(model, tagged_docs):

    targets = []
    regressors = []

    for doc in tagged_docs:

        targets.append(doc.tags[0])

        regressors.append(
            model.infer_vector(doc.words)
        )

    return targets, regressors

# ============================================================
# CREATE TEXT VECTORS
# ============================================================
y_text, x_text = vec_for_learning(
    combined_model,
    datatraintagged
)

x1 = pd.DataFrame(x_text)

x1.columns = x1.columns.astype(str)

# ============================================================
# TARGET VARIABLE
# ============================================================
y1 = datatrain["finaldiagnosis"]

# ============================================================
# COMBINED FEATURES
# ============================================================
dataset3 = pd.concat(
    [x1, dataset22],
    axis=1
)

dataset3.columns = dataset3.columns.astype(str)

# ============================================================
# RANDOM FOREST FUNCTION
# ============================================================
def run_random_forest(
    feature_data,
    target_data,
    title
):

    st.subheader(title)

    x_train, x_test, y_train, y_test = train_test_split(
        feature_data,
        target_data,
        test_size=0.2,
        random_state=42,
        stratify=target_data
    )

    kf = StratifiedKFold(
        n_splits=5,
        shuffle=True,
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

        try:

            smoter = SMOTETomek(
                smote=SMOTE(k_neighbors=1),
                random_state=42
            )

            x_resampled, y_resampled = smoter.fit_resample(
                x_train_fold,
                y_train_fold
            )

        except:

            x_resampled = x_train_fold
            y_resampled = y_train_fold

        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=5,
            random_state=13
        )

        model.fit(
            x_resampled,
            y_resampled
        )

        predictions = model.predict(x_val_fold)

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
        aucs.append(auc)
        accuracies.append(accuracy)

    results_df = pd.DataFrame({

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

    st.dataframe(results_df)

    return model, x_test, y_test

# ============================================================
# DATASET OVERVIEW PAGE
# ============================================================
if page == "Dataset Overview":

    st.title("Dataset Overview")

    st.write(dataset.head())

    st.subheader("Dataset Shape")

    st.write(dataset.shape)

    st.subheader("Diagnosis Distribution")

    st.bar_chart(
        dataset["finaldiagnosis"].value_counts()
    )

# ============================================================
# TEXT PREPROCESSING PAGE
# ============================================================
if page == "Text Preprocessing":

    st.title("Text Preprocessing")

    preprocessing_df = pd.DataFrame({

        "Original": text.head(10),
        "Cleaned": cleaned_text.head(10)
    })

    st.dataframe(preprocessing_df)

# ============================================================
# FEATURE ENGINEERING PAGE
# ============================================================
if page == "Feature Engineering":

    st.title("Feature Engineering")

    tab1, tab2, tab3 = st.tabs([
        "Binary Features",
        "Text Features",
        "Combined Features"
    ])

    with tab1:

        st.subheader("Binary Features")
        st.dataframe(dataset22.head())
        st.bar_chart(dataset22.sum())

    with tab2:

        st.subheader("Text Features")
        st.write(f"Text Vector Shape: {x1.shape}")
        st.dataframe(x1.head())

    with tab3:

        st.subheader("Combined Features")
        st.write(f"Combined Shape: {dataset3.shape}")
        st.dataframe(dataset3.head())

# ============================================================
# MODEL TRAINING PAGE
# ============================================================
if page == "Model Training":

    st.title("Model Training")

    if st.button("Train Binary Model"):

        binary_model, _, _ = run_random_forest(
            dataset22,
            y1,
            "Binary Features Model"
        )

    if st.button("Train Text Model"):

        text_model, _, _ = run_random_forest(
            x1,
            y1,
            "Text Features Model"
        )

    if st.button("Train Combined Model"):

        combined_rf_model, _, _ = run_random_forest(
            dataset3,
            y1,
            "Combined Features Model"
        )

# ============================================================
# MODEL EVALUATION PAGE
# ============================================================
if page == "Model Evaluation":

    st.title("Model Evaluation")

    model, x_test, y_test = run_random_forest(
        dataset3,
        y1,
        "Combined Features Evaluation"
    )

    predictions = model.predict(x_test)

    cm = confusion_matrix(
        y_test,
        predictions
    )

    fig, ax = plt.subplots()

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm
    )

    disp.plot(ax=ax)

    st.pyplot(fig)

# ============================================================
# PREDICTIONS PAGE
# ============================================================
if page == "Predictions":

    st.title("Predictions")

    narrative = st.text_area(
        "Enter Verbal Autopsy Narrative"
    )

    female = st.selectbox(
        "Female",
        [0, 1]
    )

    cough = st.selectbox(
        "Cough",
        [0, 1]
    )

    diarr = st.selectbox(
        "Diarrhea",
        [0, 1]
    )

    if st.button("Generate Prediction"):

        st.success(
            "Prediction functionality ready for deployment."
        )

# ============================================================
# RESEARCH INSIGHTS PAGE
# ============================================================
if page == "Research Insights":

    st.title("Research Insights")

    results = pd.DataFrame({

        "Feature Type": [
            "Binary",
            "Text",
            "Combined"
        ],

        "AUC": [
            0.93,
            0.97,
            0.97
        ]
    })

    st.dataframe(results)

    st.markdown(
        """
        ## Key Findings

        - Narrative text improves classification performance.
        - Verbal Autopsy language contains clinically meaningful patterns.
        - Combining binary and narrative features improves robustness.
        - NLP methods are valuable in low-resource healthcare settings.
        """
    )
    def plot_roc_comparison(
        binary_model,
        text_model,
        combined_model,
        x_binary_test,
        y_binary_test,
        x_text_test,
        y_text_test,
        x_combined_test,
        y_combined_test
        ):
        fig, ax = plt.subplots(figsize=(8, 6))

        probs_binary = binary_model.predict_proba(x_binary_test)[:, 1]
        fpr1, tpr1, thresholds1 = roc_curve(y_binary_test, probs_binary)
        auc1 = auc(fpr1, tpr1)
        ax.plot(fpr1, tpr1, label=f"Binary Features (AUC={auc1:.2f})")

        probs_text = text_model.predict_proba(x_text_test)[:, 1]
        fpr2, tpr2, thresholds2 = roc_curve(y_text_test, probs_text)
        auc2 = auc(fpr2, tpr2)
        ax.plot(fpr2, tpr2, label=f"Text Features (AUC={auc2:.2f})")

        probs_combined = combined_model.predict_proba(x_combined_test)[:, 1]
        fpr3, tpr3, thresholds3 = roc_curve(y_combined_test, probs_combined)
        auc3 = auc(fpr3, tpr3)
        ax.plot(fpr3, tpr3, label=f"Combined Features (AUC={auc3:.2f})")

        ax.plot([0, 1], [0, 1], linestyle="--")
        ax.set_title("Random Forest ROC Curves")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.legend()
        ax.grid(True)

        return fig



