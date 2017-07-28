from funcy import pluck
from lime.lime_text import LimeTextExplainer
import numpy as np
import pandas as pd

from models import results_df
from preprocessing import get_spacy_parser


def random_sample(pool, n):
    """Get random examples from the pool"""
    return np.random.choice(pool, n, replace=False)


def add_prediction_info(pipeline, data):
    """Add entry for predictions and entropy to data"""
    prediction_df = results_df(pipeline, pluck('content', data))
    for (row, predictions) in zip(data, prediction_df.to_dict(orient='records')):
        row.update(predictions)
    return data


def add_lime_explanation(pipeline, rows, n_classes, num_features=5, num_samples=1000):
    """Add LIME importance values to each row of data
    Args:
        pipeline: Pre-trained sklearn pipeline that takes spacy docs
        row: Dictionary with spacy to add importances to
        n_classes: Get importances for the n_classes most likely classes
        num_features: Number of features to highlight in each example
        num_samples: Number of samples to generate for each example
    """
    explainer = LimeTextExplainer(kernel_width=25,
                                  verbose=False,
                                  class_names=None,
                                  feature_selection='auto',
                                  split_expression=r'\W+',)
    spacy_parser = get_spacy_parser()

    def predict_proba(docs):
        parsed_docs = [spacy_parser(unicode(doc)) for doc in docs]
        return pipeline.predict_proba(parsed_docs)

    for row in rows:
        space_separated_tokens = ' '.join(map(str, row['content']))
        row['lime_explanation'] = explainer.explain_instance(space_separated_tokens, predict_proba,
                                                             num_features=num_features,
                                                             num_samples=num_samples,
                                                             top_labels=n_classes)
    return rows


def get_misclassified_important_tokens(data):
    """Returns a dataframe the tokens that affected incorrect classifier decisions"""
    misclassified_explanations = []
    for row in data:
        if row['label'] != row['predicted']:
            misclassified_explanations += row['lime_explanation'].as_list(label=row['predicted'])
    return pd.DataFrame(misclassified_explanations, columns=['word', 'value'])
