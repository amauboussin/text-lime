import numpy as np
import pandas as pd

from scoring import add_prediction_info, add_lime_explanation

def random_sampling(k, predict_proba, original_data, pool):
    """Random sample without replacement"""
    return np.random.choice(pool, k, replace=False)


def uncertainty_sampling(k, predict_proba, original_data, pool, dataset):
    """Choose the k examples whose predicted probabilities have the maximum entropy"""
    pool = add_prediction_info(predict_proba, pool)
    entropy_sorted_pool = sorted(pool, key=lambda row: row['entropy'], reverse=True)
    return entropy_sorted_pool[:k]


def certainty_sampling(k, predict_proba, original_data, pool, dataset):
    """Choose the k examples whose predicted probabilities have the minimum entropy"""
    pool = add_prediction_info(predict_proba, pool)
    entropy_sorted_pool = sorted(pool, key=lambda row: row['entropy'], reverse=False)
    return entropy_sorted_pool[:k]


def lime_score_sampling(k, predict_proba, original_data, pool, dataset):
    """Choose the k examples are the least well explained by a local sparse regression"""
    pool = add_lime_explanation(predict_proba, pool, 2)
    lime_score_sorted = sorted(pool, key=lambda row: row['lime_explanation'].score)
    return lime_score_sorted[:k]


def glove_mean_of_misclassified_examples(k, predict_proba, original_data, pool, dataset):
    """Choose examples that have similar document means to misclassified examples"""
    pass


def misclassified_lime_word_sampling(k, predict_proba, original_data, pool, dataset):
    """Choose the examples by looking at the most important words in misclassified examples"""
    original_data = add_lime_explanation(predict_proba, original_data, 2)
    pool = add_lime_explanation(predict_proba, pool, 2)


    token_stats = _get_lime_token_misclassification_stats(original_data)

    # filter out those that don't contribute to misclassification
    token_stats = (token_stats[token_stats.total_contribution > 0]
                   .sort_values('total_contribution', ascending=False))


    # iterate through pool, taking examples where misclassified words are important


    return None


def _get_lime_token_misclassification_stats(data):
    """Return a dataframe with token counts and contributions to misclassification decisions
    Args:
        data: List of dicts with "lime_explanation" key
    Returns:
        Dataframe with mean token contributions for_wrong_class, against_correct_class,
        total (the sum of the two), and the token_count.
    """

    if not all(['lime_explanation' in row for row in data]):
        raise ValueError('Original data doesn\'t have LIME scores')

    for_wrong_class = []
    against_correct_class = []
    for row in data:
        if row['label'] != row['predicted']:
            for_wrong_class += row['lime_explanation'].as_list(label=row['predicted'])
            against_correct_class += row['lime_explanation'].as_list(label=row['label'])

    token_values = (pd.DataFrame(for_wrong_class, columns=['token', 'for_wrong_class'])
                    .merge(pd.DataFrame(against_correct_class, columns=['token', 'against_correct_class'])))
    token_values['token'] = token_values['token'].str.lower()
    token_values['total_contribution'] = (token_values.for_wrong_class +
                                          (-1. * token_values.against_correct_class))

    token_count = token_values.groupby('token').for_wrong_class.count().to_frame('token_count')
    return token_values.groupby('token').mean().join(token_count)
