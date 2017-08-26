import numpy as np
import pandas as pd

"""
Functions to inspect the coefficients and classification results of models.
"""

def get_coef_df(model, feature_labels, class_names=None):
    """Get a dataframe of labeled coefficients from a vectorizer and fitted linear model
    Args:
        model: Sklearn estimator with coef_ property
        feature_labels: Dictionary from feature name to feature index (vectorizer.vocab_)
        class_names: Optional dictionary from label # to readable label
    Returns:
        Dataframe with columns (class, feature, and coef) with the coefficients for each class.
    """
    if not hasattr(model, 'coef_'):
        raise ValueError('Model doesn\'t have coef_ property (make sure it has been fit).')
    class_names = class_names or {}
    labels_with_coefs = get_label_with_coefs(model)

    coef_df = pd.DataFrame()
    for label in labels_with_coefs:
        class_name = class_names.get(label, label)
        class_coefs = model.coef_[label]
        feature_coef_label = [(class_name, word, class_coefs[i])
                            for word, i in feature_labels.items()]
        coef_df = coef_df.append(pd.DataFrame(feature_coef_label,
                                              columns=['class', 'feature', 'coef']))
    return coef_df


def stack_class_coefs(coef_df):
    """Reshape a dataframe of coefficients to have one row per feature
    Args:
        coef_df: Dataframe with columns (class, feature, coef)
    Returns:
        Dataframe with columns (feature, coef_1, coef_2 ...)
    """
    required_cols = ['class', 'feature', 'coef']
    if not all([required_col in coef_df.columns for required_col in required_cols]):
        raise ValueError('Columns {} not all found. (Available columns: {}'
                         .format(required_cols, coef_df.columns))

    reshaped = (coef_df.set_index(['feature', 'class']).stack()
                .to_frame('coef').unstack('class').reset_index(level=1, drop=True))
    reshaped.columns = map(lambda (a, b): '{}_{}'.format(a, b), reshaped.columns)
    return reshaped.reset_index()


def get_coef_times_value(model, X, feature_labels, class_names=None):
    """For each example in X, get the coefficient times the value in that example
    Args:
        model: Sklearn estimator with coef_ property
        X: Numpy array of feature values (in the format the model takes)
        feature_labels: Dictionary from feature name to feature index (vectorizer.vocab_)
        class_names: Optional dictionary from label # to readable label
    Returns:
        Dataframe with index (example_number, label, token) and column importance
        with values are token_coefficient * value for the given (example, class_label) pair
    """
    if not hasattr(model, 'coef_'):
        raise ValueError('Model doesn\'t have coef_ property (make sure it has been fit).')
    class_names = class_names or {}

    labels_with_coefs = get_label_with_coefs(model)
    all_values = [np.multiply(coefs, X.toarray()) for coefs in model.coef_]
    word_lookup = {index: word for word, index in feature_labels.items()}

    results_df = pd.DataFrame()
    for label, values in zip(labels_with_coefs, all_values):

        label_df = (pd.DataFrame(values).rename(columns=word_lookup.get)
                    .reset_index()
                    .rename(columns={'index': 'example_number'})
                    .assign(label=class_names.get(label, label)))
        results_df = results_df.append(label_df)
    return (results_df.set_index(['example_number', 'label'])
            .stack().to_frame('importance'))


def results_df(predict_proba, docs, labels=None):
    """Get dataframe with predicted class probabilities"""
    # make sure labels are zero-indexed
    probs = predict_proba(docs)
    n_probabilities_in_output = probs.shape[1]

    if n_probabilities_in_output < 2:
        raise ValueError('predict_proba must output a probability for each class')

    probs_df = pd.DataFrame(probs)
    predicted_class = probs_df.idxmax(axis=1)
    entropy = -np.sum(probs * np.log(probs), axis=1)

    if labels is not None:
        error = [1. - row[label]
                 for (i, row), label in zip(probs_df.iterrows(), labels)]

    if labels is not None:
        probs_df = probs_df.assign(error=error, label=labels, correct=predicted_class == labels)

    probs_df = probs_df.assign(predicted=predicted_class, entropy=entropy)
    return probs_df


def get_label_with_coefs(model):
    """Get the set of class labels that have their own coefficients"""
    n_coef_sets = model.coef_.shape[0]
    # binary classfication models only have coefs for class one
    if n_coef_sets == 1:
        return [0]
    # multiclass has coefs for each class
    else:
        return range(n_coef_sets)
