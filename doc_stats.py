import numpy as np

from models.model_utils import get_coef_df, get_bow_logistic, get_tfidf_logistic


def top_k_vector_similarity(query_docs, candidate_docs, k=5):
    """Given a list of docs, get the top k closest docs from the candidate_docs"""

    for d1 in query_docs:
        similarities = np.array([d1['content'].similarity(d2['content'])
                                 for d2 in candidate_docs])
        top_doc_indices = np.argpartition(similarities, -k)[-k:]
        ids = [doc['id'] for doc in np.array(candidate_docs)[top_doc_indices]]
        d1['similar_doc_ids'] = ids

    return query_docs


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

