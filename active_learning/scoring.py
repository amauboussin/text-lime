from funcy import pluck
from lime.lime_text import LimeTextExplainer

from models.model_utils import results_df
from preprocessing import get_spacy_parser

"""
Compute metrics for each example to be used in active learning strategies
"""


def add_prediction_info(predict_proba, data):
    """Add entry for predictions and entropy to data"""
    prediction_df = results_df(predict_proba, pluck('content', data))
    for (row, predictions) in zip(data, prediction_df.to_dict(orient='records')):
        row.update(predictions)
    return data


def add_lime_explanation(predict_proba, rows, n_classes, num_features=5, num_samples=1000):
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

    for row in rows:
        if 'lime_explanation' not in row:
            space_separated_tokens = ' '.join(map(str, row['content']))
            row['lime_explanation'] = explainer.explain_instance(space_separated_tokens,
                                                                 predict_proba_from_text,
                                                                 num_features=num_features,
                                                                 num_samples=num_samples,
                                                                 top_labels=n_classes)
    return rows
