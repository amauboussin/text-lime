from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline

from models.feature_engineering import DocsToBagOfWords, DocsToTfidf

"""Sklearn text classification pipelines that take spacy docs as input"""


def get_bow_logistic(vectorizer_params=None, clf_params=None):
    """Get a bag-of-words to logistic regression model"""
    vectorizer_params = vectorizer_params or {}
    clf_params = clf_params or {}
    return Pipeline([
            ('vectorizer', DocsToBagOfWords(**vectorizer_params)),
            ('clf', LogisticRegression(**clf_params))
        ])


def get_tfidf_logistic(vectorizer_params={}, clf_params={}):
    """Get a bag-of-words to logistic regression model"""
    vectorizer_params = vectorizer_params or {}
    clf_params = clf_params or {}
    return Pipeline([
            ('vectorizer', DocsToTfidf(**vectorizer_params)),
            ('clf', LogisticRegression(**clf_params))
        ])


def get_tfidf_svm(vectorizer_params={}, clf_params={}):
    """Get a bag-of-words to SVM model"""
    vectorizer_params = vectorizer_params or {}
    clf_params = clf_params or {}
    return Pipeline([
            ('vectorizer', DocsToTfidf(**vectorizer_params)),
            ('clf', SVC(**clf_params))
        ])


def get_tfidf_random_forest(vectorizer_params={}, clf_params={}):
    """Get a bag-of-words to logistic regression model"""
    vectorizer_params = vectorizer_params or {}
    clf_params = clf_params or {}
    return Pipeline([
            ('vectorizer', DocsToTfidf(**vectorizer_params)),
            ('clf', RandomForestClassifier(**clf_params))
        ])


def get_bow_random_forest(vectorizer_params={}, clf_params={}):
    """Get a bag-of-words to logistic regression model"""
    vectorizer_params = vectorizer_params or {}
    clf_params = clf_params or {}
    return Pipeline([
            ('vectorizer', DocsToBagOfWords(**vectorizer_params)),
            ('clf', RandomForestClassifier(**clf_params))
        ])

