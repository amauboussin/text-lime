from collections import defaultdict
import json
import os
from random import randint

import numpy as np

from active_learning.scoring import add_prediction_info
from active_learning.experiment_setup import ActiveLearningExperiment
from doc_stats import top_k_token_tag_similarity, top_k_custom_vector_similarity, top_k_glove_vector_similarity
from explain import get_confusion_matrix
from models.sklearn_models import get_bow_logistic


model = get_bow_logistic(vectorizer_params={'max_df': .8, 'min_df': .001},
                         clf_params={'C': .03})

SIMILARITY_METRICS = {
    'token_tag': top_k_token_tag_similarity,
    'glove': top_k_glove_vector_similarity,
    'custom': top_k_custom_vector_similarity,
}

N_SIMILAR = 10


#  add word importances

class ModelVisualization(object):
    def __init__(self, name, dataset, valid_percent, model_fitting_func, seed=None):

        self.name = name
        self.dataset = dataset
        self.seed = seed or randint(1, 10e9)
        self.al_experiment = ActiveLearningExperiment(dataset, model_fitting_func, None,
                                                      pool_fractions=[1.], sample_fractions=[None, 1.],
                                                      test_frac=valid_percent, seed=seed)
        self.train_data = self.al_experiment.train_pools[0]
        self.valid_data = self.al_experiment.test_pools[0]

        self.train_data_by_label = self._group_examples_by_label(self.train_data)
        self.valid_data_by_label = self._group_examples_by_label(self.train_data)
        self.confusion_matrix = None

    def fit_model(self):
        predict_proba = self.al_experiment.fit_initial_model()
        self.train_data = add_prediction_info(predict_proba, self.train_data)
        self.valid_data = add_prediction_info(predict_proba, self.valid_data)
        self.confusion_matrix = get_confusion_matrix(self.valid_data)

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

    def serialize(self, path=''):
        """Write serialized model to the given filename"""
        valid_data_to_serialize = self._prepare_examples_for_json(self.valid_data)
        confusion_matrix = self._serializable_confusion_matrix()
        data = {'examples': valid_data_to_serialize,
                'confusion_matrix': confusion_matrix}
        json.dump(data, open(os.path.join(path, '{}.json'.format(self.name)), 'w'), indent=2)

    @staticmethod
    def _prepare_examples_for_json(data):
        """Strip out unneccesary data in examples and turn spacy docs to text"""
        serialized_data = []
        for example in data:
            example_to_serialize = {}
            for key, val in example.items():
                if key == 'content':
                    val = val.text
                elif key == 'embedding_mean':
                    continue
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


