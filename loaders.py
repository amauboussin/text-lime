import csv
from random import randrange
import os

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_20newsgroups
from unidecode import unidecode

"""
Functions to load datasets from whatever format they are in to a list of dicts:
    [
        {
            id: integer id of example, (unique for each data set)
            content: "text content of example",
            label: integer class label,
            any other metadata besides the label, like pub_date, author, etc.
        }, ...
    ]
"""


def load_newsgroups():
    ng = fetch_20newsgroups()
    ids = np.arange(ng.target.size)
    data = [
        {
            'id': _id,
            'label': label,
            'content': text
        }
        for _id, label, text in zip(ids, ng.target, ng.data)
        ]
    return data


def load_ag_news(path):
    """Load ag news data from train.csv, test.csv, and labels.txt"""
    # 0 index labels
    row_to_label = lambda row: int(row[0]) - 1.
    # some of the article descriptions have \ instead of a space in some places
    # spacy requires unicode but has there is a bug serializing some unicode characters
    # unidecode works around the issue by converting to more commonly used characters
    row_to_content = lambda row: unicode(unidecode(unicode(' '.join(row[1:]).replace('\\', ' '))))
    return _load_csv_dataset(row_to_content, row_to_label, path)


def load_test_data(size, n_classes):
    """Simple synethetic data for testing"""
    return [{
        'id': i,
        'content': u'Test document {}'.format(i),
        'label': randrange(n_classes),
        'metadata': u'Test metadata {}'.format(i)
    } for i in range(1, size+1)]


def load_arxiv():
    def get_latest_archive_id():
        files = os.listdir('/Users/amauboussin/Desktop/old_arxiv/miles_archive/')
        return max([int(f.split('.')[0]) for f in files if f.endswith('pickle')])

    def strip_version(link):
        return link[:-2]

    def load():
        path_template = '/Users/amauboussin/Desktop/old_arxiv/miles_archive/{}.pickle'
        df = pd.DataFrame()
        for i in range(get_latest_archive_id() + 1):
            df = df.append(pd.read_pickle(path_template.format(i)))
        df['unique_link'] = df.link.apply(strip_version)
        return df.groupby('unique_link').first().reset_index()

    df = load()

    categories = ['cs.AI', 'cs.DS', 'stat.ML', 'cs.CL']
    # categories = ['cs.AI', 'cs.CV', 'stat.ML', 'cs.CL']

    examples = []
    _id = 1
    for label, label_name in enumerate(categories):
        class_df = df[df['category'] == label_name].sort_values('published').tail(500)
        for i, row in class_df.iterrows():
            examples.append({
                'id': _id,
                'content': unicode((row.title + ' ' + row.summary).replace('\n', ' ')),
                'published': row.published,
                'label': label
            })
            _id += 1
    return examples


def _load_sklearn_dataset():
    """Load one of the datasets included in the sklearn library"""
    pass


def _load_csv_dataset(row_to_content, row_to_label,
                      dataset_path, train_filename='train.csv', test_filename='test.csv'):
    """
    Load data that has separate csv files for the train and test examples
    Args:
        dataset_path: Folder with test, train and (optionally) class labels
        train_filename: Filename of csv with training examples
        test_filename: Filename of csv with test examples
    Returns:
        List of dicts. Keys include content, label, and any additional metadata
    """

    def get_rows(filepath, start_with_id):
        """Get label and content fields from a csv file"""
        reader = csv.reader(open(filepath, 'r'))
        return [
            {
                'id': i + start_with_id,
                'label': row_to_label(row),
                'content': row_to_content(row),
            }
            for i, row in enumerate(reader)
        ]

    #  we will handle train-test split later; just combine all data for now
    train_data = get_rows(os.path.join(dataset_path, train_filename),
                          start_with_id=1)
    test_data = get_rows(os.path.join(dataset_path, test_filename),
                         start_with_id=train_data[-1]['id'] + 1)

    return train_data + test_data
