# SMS/Email Spam Detection Project

This project classifies text messages (SMS, emails, etc.) as either spam or ham (not spam).  It uses a machine learning approach, specifically employing a Word2Vec embedding technique and an XGBoost classifier.

## Table of Contents

* [Problem Definition](#problem-definition)
* [Data Collection](#data-collection)
* [Data Cleaning and Preprocessing](#data-cleaning-and-preprocessing)
* [Feature Extraction (Word2Vec)](#feature-extraction-word2vec)
* [Model Training](#model-training)
* [Model Evaluation](#model-evaluation)
* [Model Saving](#model-saving)
* [Conclusion](#conclusion)

## Problem Definition

The goal of this project is to accurately classify text messages as spam or ham, enabling the development of a system that can filter out unwanted messages.

## Data Collection

The dataset used for this project is assumed to be in a CSV file named `spam.csv`. The notebook uses Google Drive to access the data.  The relevant columns are 'v1' (label) and 'v2' (text message).

## Data Cleaning and Preprocessing

The data is cleaned and preprocessed in the following steps:

1.  **Drop Unnecessary Columns**: Columns 'Unnamed: 2', 'Unnamed: 3', and 'Unnamed: 4' are dropped.
2.  **Label Encoding**: The target variable 'v1' is converted to binary format ('spam' as 1 and 'ham' as 0).
3.  **Text Lowercasing**: The text messages in the 'v2' column are converted to lowercase.
4.  **Removal of Punctuation, Numbers, and Special Characters**:  All characters except letters and spaces are removed from the text messages.
5.  **Stopword Removal**: Common English stopwords (e.g., "the", "is", "in") are removed.
6.  **Tokenization**: The text messages are tokenized into individual words.
7.  **Lemmatization**: The tokens are lemmatized to reduce words to their base form.

## Feature Extraction (Word2Vec)

The preprocessed text messages are converted into numerical vectors using the Word2Vec technique.

* A Word2Vec model is trained on the tokenized messages.
* Each message is then represented as the average of the Word2Vec vectors of its constituent words.

## Model Training

The dataset is split into training and testing sets.  The following machine learning models are trained:

* Gaussian Naive Bayes
* Logistic Regression
* Support Vector Machine (SVM)
* Random Forest
* XGBoost Classifier

## Model Evaluation

The performance of each model is evaluated using accuracy and a classification report (precision, recall, F1-score).  The results from the notebook are:

* Gaussian Naive Bayes:
    * Accuracy: 0.926
    * Precision: 0.83
    * Recall: 0.88
* Logistic Regression:
    * Accuracy: 0.930
    * Precision: 0.89
    * Recall: 0.79
* SVM:
    * Accuracy: 0.947
    * Precision: 0.91
    * Recall: 0.85
* Random Forest:
    * Accuracy: 0.965
    * Precision: 0.95
    * Recall: 0.90
* XGBoost:
    * Accuracy: 0.966
    * Precision: 0.94
    * Recall: 0.91

## Model Saving

The XGBoost model and the Word2Vec model are saved for later use:

* XGBoost model is saved as `xgb_model.pkl` using pickle.
* Word2Vec model is saved as `word2vec_model.bin`.

## Conclusion

The XGBoost classifier performs the best on this imbalanced dataset, achieving the highest precision, recall, and F1-score.  Therefore, the XGBoost model is chosen for deployment.
