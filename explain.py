from collections import defaultdict
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import Ridge

from neighborhood import get_neighboring_docs, get_all_masks

LOCAL_MODEL = Ridge()


class Explanation(object):

    def __init__(self, doc, local_model_by_label, score_by_label, blackbox_probs):

        self.scores = score_by_label
        self.local_models = local_model_by_label
        self.blackbox_probs = blackbox_probs

        string_tokens = [t.text for t in doc]
        self.coef_by_token = {
            label: zip(string_tokens, [coef for coef in model.coef_])
            for label, model in self.local_models.items()
        }

    def as_list(self, label):
        """Return a list of tuples (token, contribution) for the given label"""
        if label not in self.coef_by_token:
            raise ValueError('Label {} not found, options include {}'
                             .format(label, self.coef_by_token.keys()))
        return self.coef_by_token[label]

    def blackbox_prob_stats(self, label):
        """Return statistics on the distribution blackbox model probabilities over the noisy examples"""
        return pd.Series(self.blackbox_probs[:, int(label)]).describe()

    def prediction_distribution(self):
        """Return pandas series with the distribution of predicted tokens"""
        return pd.Series(np.argmax(self.blackbox_probs, axis=1)).value_counts()


def get_explanation(dataset, doc, predict_proba, n_classes=None,
                    max_simultaneous_perturbations=2, softmax_temps=None):
    """Get explanations"""
    softmax_temps = softmax_temps or [.1, 1.]
    all_tokens = []
    all_distances = []
    for temp in softmax_temps:
        tokens, distances = get_neighboring_docs(dataset, doc, max_simultaneous_perturbations,
                                                 softmax_temp=temp)
        all_tokens.append(tokens)
        all_distances.append(distances)

    all_tokens = np.concatenate(all_tokens)
    all_distances = np.concatenate(all_distances)

    blackbox_model_prediction = predict_proba([doc])[0]
    labels_to_explain = np.argsort(-blackbox_model_prediction)[:n_classes or 1]

    blackbox_probs = predict_proba(all_tokens)

    local_model_by_label = {}
    score_by_label = {}
    for label in labels_to_explain:
        local_model = clone(LOCAL_MODEL).fit(all_distances, blackbox_probs[:, label])
        score = local_model.score(all_distances, blackbox_probs[:, label])

        local_model_by_label[label] = local_model
        score_by_label[label] = score

    explanation = Explanation(doc, local_model_by_label, score_by_label, blackbox_probs)

    return explanation


def convert_probs_to_margins(y):
    """Convert class probabilities to distances from the decision boundary"""
    if not len(y.shape) == 2:
        raise ValueError('Y must be 2d numpy array of class probabilities')

    n_examples, n_classes = y.shape

    #  get distance between each probability and the max value
    max_prob = np.max(y, axis=1)
    max_prob_matrix = np.concatenate([max_prob.reshape(-1, 1)] * n_classes, axis=1)
    distance_from_max = y - max_prob_matrix

    #  get distance between max probability and second highest probability
    non_max_entries = distance_from_max[distance_from_max != 0].reshape(y.shape[0], y.shape[1] - 1)
    distance_from_max_to_margin = -1. * np.max(non_max_entries, axis=1)

    # replace max probabilities with distances to second highest element
    distance_from_max[np.arange(len(y)), np.argmax(y, axis=1)] = distance_from_max_to_margin
    return distance_from_max


def filter_rows_by_class(data, predicted_class=None, label=None):
    """Get rows with the specified predicted_class and real label"""
    filtered_data = []
    for row in data:
        correct_class = predicted_class is None or row['predicted'] == predicted_class
        correct_label = label is None or row['label'] == label
        if correct_class and correct_label:
            filtered_data.append(row)
    return filtered_data


def get_token_misclassification_stats(data, key, predicted_class=None, label=None):
    """Return a dataframe with token counts and contributions to misclassification decisions
    Args:
        data: List of dicts with Explanation or LimeExplanation objects in the given key
        key: Key where explanation objects are stored
        predicted_class: Only return stats for examples where this class was predicted.
            If None, use all examples.
    Returns:
        Dataframe with mean token contributions for_wrong_class, against_correct_class,
        total (the sum of the two), and the token_count.
    """
    if not all([key in row for row in data]):
        raise ValueError('Some or all of the data doesn\'t have key {}'.format(key))

    for_wrong_class = []
    against_correct_class = []
    for row in data:
        correct_class = predicted_class is None or row['predicted'] == predicted_class
        correct_label = label is None or row['label'] == label
        if correct_class and correct_label and row['label'] != row['predicted']:
            # check to make sure explanation exists for the given class
            for_wrong_class += row[key].as_list(row['predicted']) if row['predicted'] in row[key].scores else []
            against_correct_class += row[key].as_list(row['label']) if row['label'] in row[key].scores else []

    token_values = (pd.DataFrame(for_wrong_class, columns=['token', 'for_wrong_class'])
                    .merge(pd.DataFrame(against_correct_class, columns=['token', 'against_correct_class'])))
    token_values['token'] = token_values['token'].str.lower()
    token_values['total_contribution'] = (token_values.for_wrong_class +
                                          (-1. * token_values.against_correct_class))

    if token_values.size == 0:  # return empty dataframe with correct columns
        return pd.DataFrame(columns=['for_wrong_class','against_correct_class',
                                     'total_contribution', 'token_count', 'frequency_rank'])

    token_count = token_values.groupby('token').for_wrong_class.count().to_frame('token_count')

    token_stats = token_values.groupby('token').mean().join(token_count)
    token_stats['frequency_rank'] = token_stats.token_count.rank(pct=True)
    return token_stats


def get_important_tokens(data, key, predicted_class=None, label=None):
    """Return a dataframe with the most important tokens for the given class
    Args:
        data: List of dicts with Explanation or LimeExplanation objects in the given key
        key: Key where explanation objects are stored
        predicted_class: Only return stats for examples where this class was predicted.
            If None, use all examples.
    Returns:
        Dataframe with mean token contributions the given class and the token_count.
    """
    token_weights = []
    for row in data:
        correct_prediction = predicted_class is None or row['predicted'] == predicted_class
        correct_label = label is None or row['label'] == label
        explanation_present = key in row and row['predicted'] in row[key].scores
        if correct_prediction and correct_label and explanation_present:
            token_weights += row[key].as_list(row['predicted'])

    token_values = pd.DataFrame(token_weights, columns=['token', 'weight'])
    token_values['token'] = token_values['token'].str.lower()

    token_count = token_values.groupby('token').weight.count().to_frame('token_count')
    token_stats = token_values.groupby('token').sum().join(token_count)

    token_stats['mean_weight'] = token_stats['weight'] / token_stats['token_count']
    return token_stats


def group_examples_by_token(data, lowercase=True):
    """Return a dictionary token: indices of examples that include that token"""
    examples_by_token = defaultdict(list)
    for i, row in enumerate(data):
        tokens = [t.text.lower() if lowercase else t.text
                  for t in row['content']]
        for t in tokens:
            examples_by_token[t].append(i)
    return examples_by_token


def get_confusion_matrix(data):
    """Given data with keys for predicted class and label keys, return 2d dict confusion matrix"""
    #  assume labels range from 0 to max_label
    n_labels = int(max([row['label'] for row in data]) + 1)

    matrix = np.zeros(shape=(n_labels, n_labels))
    for row in data:
        matrix[int(row['label']), int(row['predicted'])] += 1
    return matrix


