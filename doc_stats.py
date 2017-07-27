from funcy import pluck
from lime.lime_text import LimeTextExplainer
import numpy as np

from models import get_coef_df, get_bow_logistic, get_tfidf_logistic, results_df


def top_k_vector_similarity(query_docs, candidate_docs, k=5):
    """Given a list of docs, get the top k closest docs from the candidate_docs"""

    for d1 in query_docs:
        similarities = np.array([d1['content'].similarity(d2['content'])
                                 for d2 in candidate_docs])
        top_doc_indices = np.argpartition(similarities, -k)[-k:]
        ids = [doc['id'] for doc in np.array(candidate_docs)[top_doc_indices]]
        d1['similar_doc_ids'] = ids

    return query_docs


def get_model_probabilities(pipeline, docs):
    results_df(pipeline, pluck('content', docs), pluck('labels', docs))


def fit_linear_model(train_docs, tfidf=True, vectorizer_params=None, logistic_params=None):
    """Returns a fitted linear model"""
    if tfidf:
        pipeline = get_tfidf_logistic(vectorizer_params, logistic_params)
    else:
        pipeline = get_bow_logistic(vectorizer_params, logistic_params)
    pipeline.fit(pluck('content', train_docs), pluck('label', train_docs))
    return pipeline


def linear_model_coefs(pipeline, all_docs):
    """Add linear model coefficient data for each (token, label) 
    Args:
        pipeline: Trained pipeline. Must include clf step with coef_ property
            and vectorizer with vocabularly_ property
        train_docs: Documents to train on
        all_docs: Documents to add importances to
    Return:
        all_docs with linear model importances added
    """

    coef_df = get_coef_df(pipeline.named_steps['clf'],
                          pipeline.named_steps['vectorizer'].vocabulary_)
    labels = coef_df['class'].unique()
    coef_lookup = coef_df.set_index(['class', 'feature'])['coef'].to_dict()

    for doc in all_docs:
        doc['linear_importances'] = [
            [coef_lookup.get(label, token.lower()) for label in labels]
            for token in doc]
    return all_docs


def lime_model_importances(pipeline, docs, n_classes, num_features=5, num_samples=1000):
    """Add LIME importance values for each (token, label) pair
    Args:
        pipeline: Pre-trained sklearn pipeline that takes spacy docs
        docs: Documents to add importances to
        n_classes: Get importances for the n_classes most likely classes
        num_features: Number of features to highlight in each example
        num_samples: Number of samples to generate for each example
    """
    explainer = LimeTextExplainer(kernel_width=25,
                                  verbose=False,
                                  class_names=None,
                                  feature_selection='auto',
                                  split_expression=r'\W+',)
    predict_proba = lambda doc: pipeline.predict_proba(doc['content'])
    for doc in docs:
        space_separated_tokens = ' '.join(map(str, doc))
        doc['lime_explanation'] = explainer.explain_instance(space_separated_tokens, predict_proba,
                                                             num_features=num_features,
                                                             num_samples=num_samples,
                                                             top_labels=n_classes)

    return docs
