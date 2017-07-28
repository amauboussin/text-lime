import os

from dataset_config import DATASET_CONFIG, SAVE_LOCS
import loaders
from preprocessing import parse_content_bulk
from serialize import serialize_docs, read_docs, serialize_model, read_model

class TextDataSet(object):
    """Spacy docs, metadata, models, and model output related to a text classification data set"""

    def __init__(self, dataset_name):
        """Create the data set object"""

        if dataset_name not in DATASET_CONFIG:
            raise ValueError('Dataset {} not found, options are: '.format(
                dataset_name, DATASET_CONFIG.keys()
            ))

        config = DATASET_CONFIG[dataset_name]
        self.name = dataset_name
        self.loader = self.get_loader(config['load_function'])
        self.load_dir = config['load_dir']
        self.data_file = os.path.join(SAVE_LOCS['serialized_data'],
                                      '{}.pickle'.format(self.name))

        self.data = None

    def load_data(self, use_pickle=True):
        """Load data from its original format and parse it into spacy docs"""

        if use_pickle and os.path.isfile(self.data_file):
            print 'Loading data from pickle'
            self.data = read_docs(self.data_file)
        else:
            print 'Loading data from original source'
            raw_text_data = self.loader(self.load_dir)
            self.data = list(parse_content_bulk(raw_text_data))

    def serialize_data(self):
        """Write spacy docs and metadata to disk"""
        if self.data is None:
            raise ValueError('Data is not loaded')
        serialize_docs(self.data, self.data_file)

    @staticmethod
    def get_loader(loader_name):
        if not hasattr(loaders, loader_name):
            raise ValueError('Dataset loader "{}" not found'.format(loader_name))
        return getattr(loaders, loader_name)
