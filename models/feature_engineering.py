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
                 stop_words=False, max_df=1., min_df=1):
        """Instantiate vectorizer (see CountVectorizer docs)
        Args:
            ngram_range: The lower and upper boundary of the range of n-values,
            lowercase: If true, convert all tokens to lowercase
            stop_words: If true, remove english stop words
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
                                 remove_stopwords=stop_words),
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
                 stop_words=False, max_df=1., min_df=1):
        """Instantiate vectorizer (see TfidfVectorizer docs)
        Args:
            ngram_range: The lower and upper boundary of the range of n-values,
            lowercase: If true, convert all tokens to lowercase
            stop_words: If true, remove english stop words
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
                                 remove_stopwords=stop_words),
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
