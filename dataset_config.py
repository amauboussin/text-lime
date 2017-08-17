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
    'small_test': {
        'load_function': 'load_test_data',
        'load_args': {
            'size': 1000,
            'n_classes': 4,
        }
    }
}

ROOT_DIR = '/Users/amauboussin/Desktop/text/main'
SAVE_LOCS = {
    'serialized_data': os.path.join(ROOT_DIR, 'serialized_data'),
    'models': os.path.join(ROOT_DIR, 'serialized_models'),
    'embeddings': os.path.join(ROOT_DIR, 'embeddings'),
}
