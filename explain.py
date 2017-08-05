import numpy as np
from sklearn.base import clone
from sklearn.linear_model import Ridge

from neighborhood import get_neighboring_docs, get_all_masks

LOCAL_MODEL = Ridge()


def get_explanation(dataset, doc, predict_proba, n_labels=None,
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
    labels_to_explain = np.argsort(-blackbox_model_prediction)[:n_labels or 1]

    blackbox_probs = predict_proba(all_tokens)

    explanations_by_label = {}
    for label in labels_to_explain:
        local_model = clone(LOCAL_MODEL).fit(all_distances, blackbox_probs[:, label])
        score = local_model.score(all_distances, blackbox_probs[:, label])
        explanations_by_label[label] = Explanation(doc, local_model, label, score)

    return explanations_by_label, all_tokens, all_distances, blackbox_probs


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


class Explanation(object):

    def __init__(self, doc, local_model, label, score):

        self.label = label
        self.score = score
        #  multiply coefs by 100 just to make things more readable
        self.coef_by_token = zip([round(n * -100, 4) for n in local_model.coef_], doc)


