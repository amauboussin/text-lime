from datetime import datetime

from funcy import pluck
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import train_test_split

from models.model_utils import results_df


class ActiveLearningExperiment(object):
    """Manage the train/test splits and record keeping for an active learning experiment"""

    def __init__(self, dataset, model_fitting_func, sampling_strategy,
                 pool_fractions, sample_fractions, test_frac,
                 name=None, sort_by=None, sort_by_reverse=False, seed=1):
        """Instantiate experiment
        Args:
            dataset: TextDataSet for the experiment
            model_fitting_func: Function from data -> (predict_proba function) model
            sampling_strategy: Function from (n, data) -> the n examples selected for training
            pool_fractions: List of floats indicating pool sizes
            sample_fractions: List of floats indicating how many samples are taken from each pool
            test_frac: (float) Fraction of each pool to reserve for testing
            sort_by: (str) Key to sort by when creating pools (e.g. publication date)
            sort_by_reverse: (bool) If True, sort by descending order when creating pools
            seed: (int) Seed for random state
        """
        self.dataset = dataset
        self.name = name or self.dataset.name

        if sort_by is not None:
            self.dataset.data = sorted(self.dataset.data, key=pluck(sort_by), reverse=sort_by_reverse)
        self.train_pools, self.test_pools = create_pools(self.dataset.data, pool_fractions,
                                                         test_frac, seed)
        self.sample_from_pool = sampling_strategy
        self.fit_model = model_fitting_func

        self.sample_fractions = sample_fractions

        # initialize the training data with the first pool
        self.training_data = self.train_pools[0]
        self.model_results = pd.DataFrame()

    def run(self):
        """Run an experiment end-to-end"""
        predict_proba = self.fit_initial_model()
        for i in range(1, len(self.train_pools)):
            predict_proba = self.run_iteration(i, predict_proba)

    def fit_initial_model(self):
        """Fit a model to the original training set and record its performance"""
        self.log('Fitting initial model on {} examples'.format(len(self.training_data)))
        predict_proba = self.fit_model(self.training_data)

        self.record_model_stats(predict_proba, 0)
        self.log_accuracy(predict_proba, model_number=0)

        return predict_proba

    def run_iteration(self, iteration_index, predict_proba):
        """Select examples from the next pool, train a model, and record the results"""
        pool = self.train_pools[iteration_index]
        validation_data = self.test_pools[iteration_index-1]

        n_samples_to_take = int(self.sample_fractions[iteration_index] * len(pool))
        selected_samples = self.sample_from_pool(n_samples_to_take, predict_proba,
                                                 self.training_data, validation_data, pool,
                                                 self.dataset)
        self.log('Selecting {} examples from pool of {}'.format(n_samples_to_take, len(pool)))

        self.training_data = np.concatenate((self.training_data, np.array(selected_samples)))

        self.log('Fitting model {} on {} examples'.format(iteration_index + 1, len(self.training_data)))
        predict_proba = self.fit_model(self.training_data)

        self.record_model_stats(predict_proba, iteration_index)
        self.log_accuracy(predict_proba, model_number=iteration_index)
        return predict_proba

    def record_model_stats(self, predict_proba, model_index):
        """Add to a dataframe with data on each model's accuracy on each pool"""
        for pool_index, pool in enumerate(self.test_pools):
            prediction_df = results_df(predict_proba, pluck('content', pool), pluck('label', pool))
            prediction_df['model'] = model_index
            prediction_df['pool'] = pool_index
            self.model_results = pd.concat((self.model_results, prediction_df))

    def log_accuracy(self, predict_proba, model_number):
        """Log model accuracy"""
        train_accuracy = results_df(predict_proba, pluck('content', self.training_data),
                                    pluck('label', self.training_data)).correct.mean()
        test_accuracy = self.model_results[self.model_results.model == model_number].correct.mean()

        self.log('Train accuracy: {:.2f}'.format(train_accuracy))
        self.log('Test accuracy: {:.2f}'.format(test_accuracy))

    def serialize_model_results(self):
        """Save model performance stats to disk"""
        timestamp = datetime.now().strftime('%Y-%m-%d_%H:%M')
        output_file = '{}_{}.csv'.format(self.name, timestamp)
        self.model_results.to_csv(output_file, index=False)

    @staticmethod
    def log(msg):
        """Experiment log directs here"""
        print msg


def create_pools(data, pool_fractions, test_fraction, seed=None):
    """Divide data into pools, each with a train and test set"""
    if not np.isclose(np.sum(pool_fractions), 1.):
        raise ValueError('Total of a pool fractions ({}) must be one'.format(pool_fractions))
    split_indices = (len(data) * np.cumsum(pool_fractions[:-1])).astype(int)
    pools = np.split(data, split_indices)
    train_pools, test_pools = zip(*[train_test_split(p, test_size=test_fraction, random_state=seed)
                                    for p in pools])
    return train_pools, test_pools


def create_model_fitting_func(model):
    """Get a model fitting function from an sklearn pipeline"""

    def model_fitting_func(train_data):
        """Returns a function (example -> class probabilities) from training data"""
        sklearn_model = clone(model)
        sklearn_model.fit(pluck('content', train_data), pluck('label', train_data))
        return sklearn_model.predict_proba

    return model_fitting_func
