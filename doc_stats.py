from funcy import pluck
import numpy as np
from sklearn.metrics.pairwise import cosine_distances

from models.model_utils import get_coef_df
from models.sklearn_models import get_bow_logistic, get_tfidf_logistic


def top_k_custom_vector_similarity(query_docs, candidate_docs, k):
    """Given most similar docs by looking at custom trained vector means"""
    if not all(['embedding_mean' in example for example in np.concatenate((query_docs, candidate_docs))]):
        raise ValueError('Not all embedding means have been calculated')
    distances = cosine_distances(pluck('embedding_mean', query_docs), pluck('embedding_mean', candidate_docs))
    closest_docs = np.argsort(distances, axis=1)[:, :k]
    return closest_docs


def top_k_glove_vector_similarity(query_docs, candidate_docs, k):
    """Given most similar docs by looking at glove vector means"""
    get_glove_vector = lambda row: row['content'].vector
    distances = cosine_distances(map(get_glove_vector, query_docs), map(get_glove_vector, candidate_docs))
    closest_docs = np.argsort(distances, axis=1)[:, :k]
    return closest_docs


def top_k_token_tag_similarity(query_docs, candidate_docs, k):
    """Given most similar docs by looking at (token, tag) overlap"""
    return _get_clostest_k(query_docs, candidate_docs, k, token_tag_overlap, True)


def token_tag_overlap(d1, d2):
    """Get number of (token, tag) pairs that overlap between two documents"""
    token_tags1 = set([(t.text.lower(), t.tag_) for t in d1['content']])
    token_tags2 = set([(t.text.lower(), t.tag_) for t in d2['content']])
    return len(token_tags1.intersection(token_tags2)) / float(len(token_tags1) + len(token_tags2))


def _get_clostest_k(query_docs, candidate_docs, k, distance_function, reverse=False):
    distances = np.array([[distance_function(qd, cd) for cd in candidate_docs]
                          for qd in query_docs])
    if reverse:
        distances = -distances
    closest_docs = np.argsort(distances, axis=1)[:, :k]
    return closest_docs


def explanation_dot_product_similarity(query_docs, candidate_docs, k, explanation_key):
    """Given most similar docs by looking at tokens cited as important"""
    get_label_explanation = lambda row: row[explanation_key].as_list(row['predicted'])

    def explanation_dict_dot_product(d1, d2):
        """Dot product between explanation coefficients"""
        return np.sum([d1.get(token, 0) * v for token, v in d2.items()])

    query_explanations = map(get_label_explanation, query_docs)
    candidate_explanations = map(get_label_explanation, candidate_docs)

    distances = np.array([[explanation_dict_dot_product(qe, ce) for ce in candidate_explanations]
                          for qe in query_explanations])
    closest_docs = np.argsort(-distances, axis=1)[:, :k]
    return closest_docs


def fit_linear_model(train_docs, train_labels, tfidf=True, vectorizer_params=None, logistic_params=None):
    """Returns a fitted linear model"""
    if tfidf:
        pipeline = get_tfidf_logistic(vectorizer_params, logistic_params)
    else:
        pipeline = get_bow_logistic(vectorizer_params, logistic_params)
    pipeline.fit(train_docs, train_labels)
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

