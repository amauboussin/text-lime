from collections import Counter
from functools import partial

import numpy as np
from sklearn.base import TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from spacy.tokens.doc import Doc

"""
Sklearn transforms to go from spacy docs to formats that can be fed into a model
"""


def _get_tokens(doc_or_token_list, lowercase=True, remove_stopwords=False):
    """Return list of tokens from a spacy doc + associated metadata"""
    if type(doc_or_token_list) == Doc:
        return [t.text.lower() if lowercase else t.text
                for t in doc_or_token_list
                if not (remove_stopwords and t.is_stop)
                ]
    else: # list of tokens
        return [token.lower() if lowercase else token
                for token in doc_or_token_list]


def _identity(x):
    """Can't use a lambda for this if we want to pickle these classes"""
    return x


class DocsToBagOfWords(CountVectorizer):
    """Extend CountVectorizer to take spacy docs or token lists as input"""

    def __init__(self, ngram_range=(1, 1), lowercase=True,
                 remove_stop_words=True, max_df=1., min_df=1):
        """Instantiate vectorizer (see CountVectorizer docs)
        Args:
            ngram_range: The lower and upper boundary of the range of n-values,
            lowercase: If true, convert all tokens to lowercase
            remove_stop_words: If true, remove english stop words
            max_df: (float or int) Ignore terms that have a document frequency
                strictly higher than the given threshold.
                If float, the parameter represents a proportion of documents,
                if integer it represents absolute counts.
            min_df: (float or int) Similar to above, but a minimum threshold
        """

        #  spacy docs are already tokenized, so we can just get the tokenized list
        #  in the preprocessor and do nothing for the tokenization step
        super(DocsToBagOfWords, self).__init__(
            preprocessor=partial(_get_tokens, lowercase=lowercase,
                                 remove_stopwords=remove_stop_words),
            tokenizer=_identity,
            ngram_range=ngram_range,
            lowercase=False,
            stop_words=None,
            max_df=max_df,
            min_df=min_df
        )


class DocsToTfidf(TfidfVectorizer):
    """Extend TfidfVectorizer to take spacy docs or token lists as input"""

    def __init__(self, ngram_range=(1, 1), lowercase=True,
                 remove_stop_words=True, max_df=1., min_df=1):
        """Instantiate vectorizer (see TfidfVectorizer docs)
        Args:
            ngram_range: The lower and upper boundary of the range of n-values,
            lowercase: If true, convert all tokens to lowercase
            remove_stop_words: If true, remove english stop words
            max_df: (float or int) Ignore terms that have a document frequency
                strictly higher than the given threshold.
                If float, the parameter represents a proportion of documents,
                if integer it represents absolute counts.
            min_df: (float or int) Similar to above, but a minimum threshold
        """

        #  spacy docs are already tokenized, so we can just get the tokenized list
        #  in the preprocessor and do nothing for the tokenization step
        super(DocsToTfidf, self).__init__(
            preprocessor=partial(_get_tokens, lowercase=lowercase,
                                 remove_stopwords=remove_stop_words),
            tokenizer=_identity,
            ngram_range=ngram_range,
            lowercase=False,
            stop_words=None,
            max_df=max_df,
            min_df=min_df
        )


class DocsToGloveMean(TransformerMixin):
    """Sklearn transform that transforms spacy docs to glove vector means"""

    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        """Get mean glove vector for each doc in X"""
        return np.array([doc.vector for doc in X])


class DocToWordIndices(TransformerMixin):
    """Take"""
    n_special_chars = 2
    padding_index = 0
    unk_index = 1

    def __init__(self, max_seq_length=None, vocab_size=None, case_sensitive=False,
                 pad_to_max_length=True, left_padding=0):
        """Create sklearn transformer that goes from spacy docs to a list of word indicices 
        Args:
            max_seq_length: If not None, truncate sequences beyond this length
            vocab_size: If not None, words outside the vocab_size most common words will be UNK
            case_sensitive: If True, different capitalization maps to different tokens
            pad_to_max_length: If True, zero pad sequences so they are all the same length
            
        """
        self.case_sensitive = case_sensitive
        self.vocab_size = vocab_size
        self.token_lookup = None
        self.pad_to_max_length = pad_to_max_length
        self.left_padding = left_padding
        self.max_seq_length = max_seq_length

    def fit(self, X, y=None):
        doc_tokens = [_get_tokens(doc) for doc in X]
        self.dt = doc_tokens
        self.max_seq_length = self.max_seq_length or max(map(len, doc_tokens))
        token_counts = Counter([t for doc in doc_tokens for t in doc])
        n_tokens = self.vocab_size or sum(token_counts.itervalues())
        self.token_lookup = {token: i + self.n_special_chars
                             for i, (token, count) in enumerate(token_counts.most_common(n_tokens))}
        return self

    def _transform_doc(self, doc):
        """Transform a single spacy tokens into a list of indexed tokens"""
        doc_indices = [self.token_lookup.get(t, self.unk_index)
                       for t in _get_tokens(doc)]

        if self.pad_to_max_length and len(doc_indices) <= self.max_seq_length:
            right_padding = self.max_seq_length - len(doc_indices)
        else:
            right_padding = 0

        return np.pad(doc_indices[:self.max_seq_length],
                      (self.left_padding, right_padding),
                      mode='constant',
                      constant_values=self.padding_index).reshape(1, -1)

    def transform(self, X, y=None):
        return np.concatenate([self._transform_doc(doc) for doc in X], axis=0)
