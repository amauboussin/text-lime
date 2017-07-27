import csv
import os

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
    #fetch_20newsgroups(subset='train',
    pass


def load_ag_news(path):
    """Load ag news data from train.csv, test.csv, and labels.txt"""
    # 0 index labels
    row_to_label = lambda row: int(row[0]) - 1.
    # some of the article descriptions have \ instead of a space in some places
    # spacy requires unicode but has there is a bug serializing some unicode characters
    # unidecode works around the issue by converting to more commonly used characters
    row_to_content = lambda row: unicode(unidecode(unicode(' '.join(row[1:]).replace('\\', ' '))))
    return _load_csv_dataset(row_to_content, row_to_label, path)


def _load_sklearn_dataset():
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
