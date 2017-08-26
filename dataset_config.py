import os

DATASET_CONFIG = {
    'ag_news': {
        'name': 'ag_news',
         # function that maps from directory -> list of dicts with text and associated metadata
        'load_function': 'load_ag_news',
        # input to load_function
        'load_args': {
            'path': 'data/ag_news_csv',
        },
    },
    'arxiv': {
        'name': 'arxiv',
        'load_function': 'load_arxiv',
    },
    '20newsgroups': {
        'load_function': 'load_newsgroups',
    },
    'mr': {
        'load_function': 'load_movie_reviews',
        'load_args': {
            'path': 'data/mr/rt-polarity.all.txt',
        },
    },
    'small_test': {
        'load_function': 'load_test_data',
        'load_args': {
            'size': 1000,
            'n_classes': 4,
        }
    }
}

WEB_DIR = '/Users/amauboussin/Desktop/text/main/mmos/web'
DATA_DIR = '/Users/amauboussin/Desktop/text/main'
SAVE_LOCS = {
    'serialized_data': os.path.join(DATA_DIR, 'serialized_data'),
    'models': os.path.join(DATA_DIR, 'serialized_models'),
    'embeddings': os.path.join(DATA_DIR, 'embeddings'),
}
