from itertools import product

import numpy as np

from explain import get_token_misclassification_stats, group_examples_by_token, get_confusion_matrix
from neighborhood import softmax
from scoring import add_prediction_info, add_lime_explanation, add_mmos_explanations


def random_sampling(k, predict_proba, original_data, valid_data, pool, dataset):
    """Random sample without replacement"""
    return np.random.choice(pool, k, replace=False)


def uncertainty_sampling(k, predict_proba, original_data, valid_data, pool, dataset):
    """Choose the k examples whose predicted probabilities have the maximum entropy"""
    pool = add_prediction_info(predict_proba, pool)
    entropy_sorted_pool = sorted(pool, key=lambda row: row['entropy'], reverse=True)
    return entropy_sorted_pool[:k]


def certainty_sampling(k, predict_proba, original_data, valid_data, pool, dataset):
    """Choose the k examples whose predicted probabilities have the minimum entropy"""
    pool = add_prediction_info(predict_proba, pool)
    entropy_sorted_pool = sorted(pool, key=lambda row: row['entropy'], reverse=False)
    return entropy_sorted_pool[:k]


def explanation_variance(k, predict_proba, original_data, valid_data, pool, dataset):
    pool = add_mmos_explanations(predict_proba, pool, dataset, 1)
    pool = add_prediction_info(predict_proba, pool)

    def get_predicted_class_variance(example):
        predicted_class = example['predicted']
        return example['explanation'].blackbox_prob_stats(predicted_class)['std']

    class_variance_sorted = sorted(pool, key=get_predicted_class_variance, reverse=False)
    return class_variance_sorted[:k]


def explanation_score(k, predict_proba, original_data, valid_data, pool, dataset):
    pool = add_mmos_explanations(predict_proba, pool, dataset, 1, softmax_temps=[1.])
    pool = add_prediction_info(predict_proba, pool)

    def get_predicted_class_score(example):
        return example['explanation'].scores.values()[0]

    class_score_sorted = sorted(pool, key=get_predicted_class_score, reverse=False)
    return class_score_sorted[:k]


def lime_score_sampling(k, predict_proba, original_data, valid_data, pool, dataset):
    """Choose the k examples are the least well explained by a local sparse regression"""
    pool = add_lime_explanation(predict_proba, pool, 2)
    lime_score_sorted = sorted(pool, key=lambda row: row['lime_explanation'].score)
    return lime_score_sorted[:k]


def embedding_mean_similarity(k, predict_proba, original_data, valid_data, pool, dataset):
    """Choose examples that have similar document means to misclassified examples"""

    # all_similarities = cosine_similarity(self.ft_matrix, vector.reshape(1, -1)).squeeze()
    # most_similar_indices = (-all_similarities).argsort()[:k]
    # return self.ft_vocab[most_similar_indices]
    pass


def contains_misunderstood_words(k, predict_proba, original_data, valid_data, pool, dataset):
    """Choose examples that contain words that played a role in vaidation data misclassification"""

    softmax_temp = .025
    explanation_key = 'explanation'
    valid_data = add_mmos_explanations(predict_proba, valid_data, dataset, 1, softmax_temps=[1.])
    valid_data = add_prediction_info(predict_proba, valid_data)

    confusion_matrix = get_confusion_matrix(data)
    all_labels = range(confusion_matrix.shape[0])

    all_token_stats = {}
    confusion_matrix_counts = []
    confusion_matrix_labels = []
    for label, predicted_class in product(all_labels, all_labels):
        if label == predicted_class:
            continue
        stats = get_token_misclassification_stats(pool, explanation_key, predicted_class, label)

        #  only keep things tokens that on average contributed to misclassification
        stats = stats[stats.total_contribution > 0]
        stats['rank'] = stats.frequency_rank * stats.total_contribution
        stats['sampling_prob'] = softmax(stats['rank'].values, softmax_temp)
        all_token_stats[(label, predicted_class)] = stats

        confusion_matrix_labels.append((label, predicted_class))
        confusion_matrix_counts.append(confusion_matrix[label, predicted_class])

    confusion_matrix_probs = confusion_matrix_counts / np.sum(confusion_matrix_counts)

    samples_by_token = group_examples_by_token(valid_data)

    #  iteratively sample new examples until we have k
    selected_examples = set()
    while len(selected_examples) < k:
        # sample uniformly from non-diagonal confusion matrix cells
        bin = np.random.choice(confusion_matrix_labels, p=confusion_matrix_probs)

        # sample from token importance with softmax
        token = np.random.choice(all_token_stats[bin].index.values,
                                 p=all_token_stats[bin].sampling_prob)

        if token not in samples_by_token:
            continue
        # sample from pool containing words
        example_index = np.random.choice(samples_by_token[token])
        if example_index in selected_examples:
            continue
        else:
            selected_examples.add(example_index)

    return [valid_data[i] for i in selected_examples]




