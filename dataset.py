from collections import defaultdict
import os
import re

import fasttext
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from dataset_config import DATASET_CONFIG, SAVE_LOCS
import loaders
from preprocessing import parse_content_bulk, parse_content_serial
from serialize import serialize_docs, read_docs, serialize_model, read_model

FT_FILTER_REGEX = re.compile('[^a-zA-Z:]')


class TextDataSet(object):
    """Spacy docs, metadata, unsupervised embeddings, and model output related to a text data set"""
    def __init__(self, dataset_name, loader=None):
        """Create the data set object"""

        if dataset_name not in DATASET_CONFIG and loader is None:
            raise ValueError('Dataset {} not found, options are: '.format(
                dataset_name, DATASET_CONFIG.keys()
            ))

        config = DATASET_CONFIG.get(dataset_name, {})
        self.name = dataset_name
        self.loader = loader or self.get_loader(config['load_function'])
        self.load_args = config.get('load_args', {})
        self.data_file = os.path.join(SAVE_LOCS['serialized_data'],
                                      '{}.pickle'.format(self.name))
        self.data = None
        self.n_classes = None

        self.ft_input_file = os.path.join(SAVE_LOCS['embeddings'],
                                          '{}_ft_input.txt'.format(self.name))
        self.ft_model_file = os.path.join(SAVE_LOCS['embeddings'], self.name) + '.bin'
        self.ft_model = None
        self.ft_vocab = None
        self.ft_matrices = None
        # only tokens of the same category are eligible to be similar words
        self.get_token_category = lambda t: t.tag_

    def load_data(self, use_pickle=True):
        """Load data from its original format and parse it into spacy docs"""

        if use_pickle and os.path.isfile(self.data_file):
            print 'Loading data from pickle'
            self.data = read_docs(self.data_file)
        else:
            print 'Loading data from original source'
            raw_text_data = self.loader(**self.load_args)
            self.data = np.array(list(parse_content_serial(raw_text_data)))

        self.n_classes = len(set([row['label'] for row in self.data]))

    def shuffle_data(self):
        """Shuffle the data so it is in random order"""
        np.random.shuffle(self.data)

    def serialize_data(self):
        """Write spacy docs and metadata to disk"""
        if self.data is None:
            raise ValueError('Data is not loaded')
        serialize_docs(self.data, self.data_file)

    def fasttext_preprocess(self):
        """Write a text file that can be fed into fasttext to train unsupervised embeddings"""
        lines = [self._get_fasttext_representation(row['content']) + '\n'
                 for row in self.data]
        with open(self.ft_input_file, 'w') as wfile:
            wfile.writelines(lines)

    def load_fasttext_model(self):
        """Load serialized fast text embeddings"""
        if not os.path.isfile(self.ft_model_file):
            raise Exception('No fast text embedding model found at {}'.format(self.ft_model_file))

        self.ft_model = fasttext.load_model(self.ft_model_file)
        self._setup_vector_similarity()

    def create_embeddings(self, method='skipgram', overwrite=False):
        """Train word embeddings using fasttext"""
        if method not in ['skipgram', 'cbow']:
            raise ValueError('Method must be skipgram or cbow')
        output_name = os.path.join(SAVE_LOCS['embeddings'], self.name)
        if overwrite or (not os.path.isfile(self.ft_model_file)):
            self.fasttext_preprocess()
            if method == 'skipgram':
                fasttext.skipgram(self.ft_input_file, output_name)
            else:
                fasttext.cbow(self.ft_input_file, output_name)
        self.load_fasttext_model()

    def _setup_vector_similarity(self):
        """Setup matrix of word vectors for similarity calculations"""
        if self.ft_model is None:
            raise ValueError('Can\'t get vector similarity before loading fast text embeddings')

        #  group tokens by tag
        self.tokens_by_tag = defaultdict(set)
        for row in self.data:
            for token in row['content']:
                token_category = self.get_token_category(token)
                self.tokens_by_tag[token_category].add(token.text.lower())

        #  create a matrix of word embeddings for each tag
        self.ft_matrices = {}
        self.ft_vocab = {}
        for tag, this_tag_tokens in self.tokens_by_tag.items():
            word_vectors_and_values = [(self.ft_model[word], word)
                                       for word in self.ft_model.words
                                       if word in this_tag_tokens
                                       ]
            if len(word_vectors_and_values) == 0:
                continue
            word_matrix, word_values = zip(*word_vectors_and_values)
            self.ft_matrices[tag] = np.array(word_matrix)
            self.ft_vocab[tag] = np.array(word_values)

    def get_all_embedding_means(self, include_stopwords=False):
        """Save embedding mean for every document"""
        for row in self.data:
            row['embedding_mean'] = self.get_embedding_mean(row['content'], include_stopwords)

    def get_word_vector(self, word):
        """Get 1d numpy array of embedding for the given word"""
        if self.ft_model is None:
            raise Exception('No fast text model loaded')
        if word not in self.ft_model:
            return np.array([])
        return np.array(self.ft_model[word])

    def most_similar_to_vector(self, vector, k):
        """K nearest neighbors to the a 1d word vector by cosine distance"""
        all_similarities = cosine_similarity(self.ft_matrix, vector.reshape(1, -1)).squeeze()
        most_similar_indices = (-all_similarities).argsort()[:k]
        return self.ft_vocab[most_similar_indices]

    def most_similar_to_word(self, word, k):
        """K nearest neighbors to the given word's vector by cosine distance"""
        return self.most_similar_to_vector(self.get_word_vector(word), k)

    def get_embedding_mean(self, doc, include_stopwords=True):
        """Get mean embedding of the given document"""
        tokens = self._get_fasttext_representation(doc, include_stopwords)
        embeddings = np.array([self.ft_model[token] for token in tokens])
        return np.mean(embeddings, axis=0)

    @staticmethod
    def _get_fasttext_representation(doc, include_stopwords=True):
        """Remove non-alpha characters and convert to lowercase for input into fasttext"""
        tokens = [re.sub(FT_FILTER_REGEX, '', token.text).lower()
                  for token in doc
                  if include_stopwords or (not token.is_stop)]
        return ' '.join([token for token in tokens if token.strip()])

    @staticmethod
    def get_loader(loader_name):
        """Get function that goes from directory -> formatted dataset"""
        if not hasattr(loaders, loader_name):
            raise ValueError('Dataset loader "{}" not found'.format(loader_name))
        return getattr(loaders, loader_name)
