from collections import defaultdict
from funcy import pluck
from itertools import product
import json
from random import randint

import numpy as np
from sklearn.model_selection import train_test_split

from active_learning.scoring import add_prediction_info, add_lime_explanation, add_mmos_explanations
from doc_stats import top_k_token_tag_similarity, top_k_custom_vector_similarity, top_k_glove_vector_similarity
from explain import get_confusion_matrix, get_important_tokens, get_token_misclassification_stats
from models.sklearn_models import get_bow_logistic


model = get_bow_logistic(vectorizer_params={'max_df': .8, 'min_df': .001},
                         clf_params={'C': .03})

SIMILARITY_METRICS = {
    'token_tag': top_k_token_tag_similarity,
    'glove': top_k_glove_vector_similarity,
    'custom': top_k_custom_vector_similarity,
}

SOFTMAX_TEMPS = [1., 3.,  5.]
SOFTMAX_TEMPS = [.1, .3,  .8]
SOFTMAX_TEMPS = None

N_LIME_FEATURES = 8
N_LIME_SAMPLES = 10

N_SIMILAR = 10


class ModelVisualization(object):

    def __init__(self, name, dataset, train_data, valid_data, predict_proba, class_labels):
        self.dataset = dataset
        self.name = name
        self.class_labels = class_labels or map(str, range(self.dataset.n_classes))
        self.predict_proba = predict_proba
        self.train_data = train_data
        self.valid_data = valid_data

        self.train_data_by_label = self._group_examples_by_label(self.train_data)
        self.valid_data_by_label = self._group_examples_by_label(self.train_data)

        self.confusion_matrix = None
        self.explanation_aggregates = None

    @classmethod
    def from_sklearn_model(cls, name, dataset, valid_percent, model, split_random_seed=None, labels=None):

        split_random_seed = split_random_seed or randint(0, 2**32-1)
        train_data, valid_data = train_test_split(dataset.data, test_size=valid_percent,
                                                  random_state=split_random_seed)
        model.fit(pluck('content', train_data), pluck('label', train_data))
        return cls(name, dataset, train_data, valid_data, model.predict_proba, labels)

    def get_predictions(self):
        self.train_data = add_prediction_info(self.predict_proba, self.train_data)
        self.valid_data = add_prediction_info(self.predict_proba, self.valid_data)
        self.confusion_matrix = get_confusion_matrix(self.valid_data)

    def add_lime_explanations(self):
        #  add explanation. aggregate for each confusion matrix cell
        self.valid_data = add_lime_explanation(self.predict_proba, self.valid_data,
                                               self.dataset.n_classes,
                                               num_features=N_LIME_FEATURES,
                                               num_samples=N_LIME_SAMPLES)
        self.lime_aggregates = self._aggregate_reasons(self.valid_data, 'lime_explanation')

        # swap out explanation to just explain the predicted class
        for row in self.valid_data:
            row['lime_explanation'] = row['lime_explanation'].as_list(int(row['predicted']))
            row['lime_explanation'] = [(token, round(value, 4)) for token, value in row['lime_explanation']]

    def add_explanations(self, set_size=2, softmaxt_temps=None):
        softmaxt_temps = softmaxt_temps or SOFTMAX_TEMPS
        self.valid_data = add_mmos_explanations(self.predict_proba, self.valid_data, self.dataset,
                                                self.dataset.n_classes, softmax_temps=softmaxt_temps,
                                                max_simultaneous_perturbations=set_size)
        self.explanation_aggregates = self._aggregate_reasons(self.valid_data, 'explanation')

        # swap out explanation to just explain the predicted class
        for row in self.valid_data:
            row['original_explanation'] = row['explanation']
            row['explanation'] = row['explanation'].as_list(row['predicted'])
            row['explanation'] = [(token, round(value, 4)) for token, value in row['explanation']]

    def _aggregate_reasons(self, data, explanation_key):
        """Return dictionary (label, predicted) -> Dataframe of misclassified tokens"""
        all_labels = range(self.dataset.n_classes)
        aggregate_reasons = {}
        for label, predicted_class in product(all_labels, all_labels):
            if label == predicted_class:
                stats = get_important_tokens(data, explanation_key,
                                             predicted_class, label)
            else:
                stats = get_token_misclassification_stats(data, explanation_key,
                                                          predicted_class, label)
            aggregate_reasons[(label, predicted_class)] = stats
        return aggregate_reasons

    def add_similarities(self):
        self.dataset.get_all_embedding_means()
        for metric_name, similarity_function in SIMILARITY_METRICS.items():
            self._get_classwise_simlarities(self.valid_data,
                                            self.train_data_by_label,
                                            metric_name,
                                            similarity_function)
            self._get_classwise_simlarities(self.valid_data,
                                            self.valid_data_by_label,
                                            metric_name,
                                            similarity_function)

    def _get_classwise_simlarities(self, query_data, data_by_label, function_name, similarity_function):
        """For each document in query_data, calculate the N_SIMILAR closest docs"""
        for label, examples in data_by_label.items():
            example_id_lookup = {i: row['id'] for i, row in enumerate(examples)}

            similar_doc_indices = similarity_function(query_data, examples, N_SIMILAR)
            all_similar_doc_ids = np.vectorize(example_id_lookup.get)(similar_doc_indices)

            for example_similar_ids, queried_example in zip(all_similar_doc_ids, query_data):
                queried_example['{}_{}_similar'.format(function_name, label)] = list(example_similar_ids)

    def _serializable_confusion_matrix(self):
        """Get json serializable version of confusion matrix"""
        if self.confusion_matrix is None:
            raise ValueError('Can\'t generate confusion matrix because model has not been fit yet.')
        return map(list, self.confusion_matrix)

    def _serialize_aggregate_importances(self, n_words_per_class=25):
        serializable_aggregate_explanations = []
        for label in range(len(self.class_labels)):
            explanation_row = []
            for predicted in range(len(self.class_labels)):
                agg_df = self.explanation_aggregates[(label, predicted)]

                if agg_df.size == 0:
                    explanation_row.append({})
                    continue
                if label == predicted:
                    top_tokens = (agg_df.sort_values('weight', ascending=False)
                                  .head(n_words_per_class).reset_index()[['token', 'weight']])
                else:
                    agg_df['rank'] = agg_df.frequency_rank * agg_df.total_contribution
                    top_tokens = (agg_df.sort_values('rank', ascending=False)
                                  .head(25).reset_index()[['token', 'total_contribution']])

                explanation_row.append(top_tokens.to_dict(orient='records'))
            serializable_aggregate_explanations.append(explanation_row)
        return serializable_aggregate_explanations

    def serialize(self, filename):
        """Write serialized model to the given filename"""
        examples_to_serialize = self._prepare_examples_for_json(self.valid_data)
        confusion_matrix = self._serializable_confusion_matrix()
        data = {
            'examples': examples_to_serialize,
            'confusion_matrix': confusion_matrix,
            'name': self.name,
            'labels': self.class_labels
        }
        if self.explanation_aggregates is not None:
            data['aggregate_explanation'] = self._serialize_aggregate_importances()
        json.dump(data, open(filename, 'w'), indent=2)

    @staticmethod
    def _prepare_examples_for_json( data):
        """Strip out unneccesary data in examples and turn spacy docs to text"""
        serialized_data = []
        for example in data:
            example_to_serialize = {}
            for key, val in example.items():
                if key == 'content':
                    val = val.text
                elif key == 'embedding_mean':
                    continue
                elif key == 'published':
                    val = str(val)
                if type(val) == np.float64:
                    val = round(val, 4)
                example_to_serialize[key] = val
            serialized_data.append(example_to_serialize)
        return serialized_data

    @staticmethod
    def _group_examples_by_label(examples):
        examples_by_label = defaultdict(list)
        for e in examples:
            examples_by_label[e['label']].append(e)
        return examples_by_label


