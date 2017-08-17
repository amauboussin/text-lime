from funcy import pluck
from lime.lime_text import LimeTextExplainer

from explain import get_explanation
from models.model_utils import results_df
from preprocessing import get_spacy_parser

"""
Compute metrics for each example to be used in active learning strategies
"""

DEFAULT_SOFTMAX_TEMPS = [.1, 1., 3.]


def add_prediction_info(predict_proba, data):
    """Add entry for predictions and entropy to data"""
    prediction_df = results_df(predict_proba, pluck('content', data))
    for (row, predictions) in zip(data, prediction_df.to_dict(orient='records')):
        row.update(predictions)
    return data


def add_mmos_explanations(predict_proba, data, dataset, n_classes, softmax_temps=None,
                          max_simultaneous_perturbations=2):
    if softmax_temps is None:
        softmax_temps = DEFAULT_SOFTMAX_TEMPS
    for i, row in enumerate(data):
        if i % 1000 == 0:
            print 'Done with ', i
        row['explanation'] = get_explanation(dataset, row['content'], predict_proba,
                                             n_classes=n_classes,
                                             max_simultaneous_perturbations=max_simultaneous_perturbations,
                                             softmax_temps=softmax_temps)

    return data


def add_lime_explanation(predict_proba, data, n_classes, num_features=8, num_samples=1000):
    """Add LIME importance values to each row of data
    Args:
        predict_proba: Function that goes from example -> class probabilities
        row: Dictionary with spacy docs
        n_classes: Get importances for the n_classes most likely classes
        num_features: Number of features to highlight in each example
        num_samples: Number of samples to generate for each example
        overwrite: Recompute explanations if they are already in place
    """
    explainer = LimeTextExplainer(kernel_width=25,
                                  verbose=False,
                                  class_names=None,
                                  feature_selection='auto',
                                  split_expression=r'\W+',)
    spacy_parser = get_spacy_parser()

    def predict_proba_from_text(docs):
        """Modify predict_proba to take raw strings instead of spacy docs"""
        parsed_docs = [spacy_parser(unicode(doc)) for doc in docs]
        return predict_proba(parsed_docs)

    for row in data:
        if True or 'lime_explanation' not in row:
            space_separated_tokens = ' '.join(map(str, row['content']))
            row['lime_explanation'] = explainer.explain_instance(space_separated_tokens,
                                                                 predict_proba_from_text,
                                                                 num_features=num_features,
                                                                 num_samples=num_samples,
                                                                 top_labels=n_classes)
    return data
