import os

DATASET_CONFIG = {
    'ag_news': {
        'name': 'ag_news',
         # function that maps from directory -> list of dicts with text and associated metadata
        'load_function': 'load_ag_news',
        # input to load_function
        'load_dir': 'data/ag_news_csv',
    }

}
ROOT_DIR = '/Users/amauboussin/Desktop/text/main'
SAVE_LOCS = {
    'serialized_data': os.path.join(ROOT_DIR, 'serialized_data'),
    'models': os.path.join(ROOT_DIR,'serialized_models'),
}



