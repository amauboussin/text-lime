import os
import re

import fasttext
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from dataset_config import DATASET_CONFIG, SAVE_LOCS
import loaders
from preprocessing import parse_content_bulk
from serialize import serialize_docs, read_docs, serialize_model, read_model

FT_FILTER_REGEX = re.compile('[^a-zA-Z:]')


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

        self.ft_input_file = os.path.join(SAVE_LOCS['embeddings'],
                                          '{}_ft_input.txt'.format(self.name))
        self.ft_model_file = os.path.join(SAVE_LOCS['embeddings'], self.name) + '.bin'
        self.ft_model = None
        self.ft_vocab = None
        self.ft_matrix = None

    def load_data(self, use_pickle=True):
        """Load data from its original format and parse it into spacy docs"""

        if use_pickle and os.path.isfile(self.data_file):
            print 'Loading data from pickle'
            self.data = read_docs(self.data_file)
        else:
            print 'Loading data from original source'
            raw_text_data = self.loader(self.load_dir)
            self.data = np.array(list(parse_content_bulk(raw_text_data)))

    def serialize_data(self):
        """Write spacy docs and metadata to disk"""
        if self.data is None:
            raise ValueError('Data is not loaded')
        serialize_docs(self.data, self.data_file)

    def fasttext_preprocess(self):
        """Write a text file that can be fed into fasttext to train unsupervised embeddings"""
        lines = [self.get_fasttext_representation(row['content']) + '\n'
                 for row in self.data]
        with open(self.ft_input_file, 'w') as wfile:
            wfile.writelines(lines)

    def load_fasttext_model(self):
        """Load serialized fast text embeddings"""
        if not os.path.isfile(self.ft_model_file):
            raise Exception('No fast text embedding model found at {}'.format(self.ft_model_file))

        self.ft_model = fasttext.load_model(self.ft_model_file)

    def create_embeddings(self, overwrite=False):
        """Train word embeddings using fasttext"""
        output_name = os.path.join(SAVE_LOCS['embeddings'], self.name)
        if overwrite or (not os.path.isfile(self.ft_model_file)):
            fasttext.skipgram(self.ft_input_file, output_name)

    def _setup_vector_similarity(self):
        """Setup matrix of word vectors for similarity calculations"""
        if self.ft_model is None:
            raise ValueError('Can\'t get vector similarity before loading fast text embeddings')
        self.ft_vocab = np.array(list(self.ft_model.words))
        self.ft_matrix = np.array([self.ft_model[word] for word in self.ft_vocab])

    def get_word_vector(self, word):
        """Get 1d numpy array of embedding for the given word"""
        if self.ft_model is None:
            raise Exception('No fast text model loaded')
        if word not in self.ft_model:
            return np.array([])
        return np.array(self.ft_model[word])

    def most_similar_to_vector(self, vector, k):
        """K nearest neighbors to the a 1d word vector by cosine distance"""
        if self.ft_matrix is None:
            self._setup_vector_similarity()
        all_similarities = cosine_similarity(self.ft_matrix, vector.reshape(1, -1))
        most_similar_indices = (-all_similarities).argsort()[:k]
        return self.ft_vocab[most_similar_indices]

    def most_similar_to_word(self, word, k):
        """K nearest neighbors to the given word's vector by cosine distance"""
        return self.most_similar_to_vector(self.get_word_vector(word), k)

    @staticmethod
    def get_fasttext_representation(doc):
        """Remove non-alpha characters and convert to lowercase for input into fasttext"""
        tokens = [re.sub(FT_FILTER_REGEX, '', token.text).lower() for token in doc]
        return ' '.join([token for token in tokens if token.strip()])

    @staticmethod
    def get_loader(loader_name):
        if not hasattr(loaders, loader_name):
            raise ValueError('Dataset loader "{}" not found'.format(loader_name))
        return getattr(loaders, loader_name)
