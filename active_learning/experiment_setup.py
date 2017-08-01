
class ActiveLearningExperiment(object):
    """Manage the train/test splits and record keeping for an active learning experiment"""

    def __init__(self, model, initial_set_size, iterations=1, sort_by=None, seed=None):
        """Instantiate experiment
        Args:
            initial_set_size: (float) proportion of data in initial training_set
            iterations: (int) number of times to select from the pool
            sort_by: (function) Key to sort by when creating pools (e.g. publication date)
            seed: (int) Seed for random state
        """

        # lists of indices
        self.initial_set, self.pool =
        self.pools_train = []
        self.pools_test = []
