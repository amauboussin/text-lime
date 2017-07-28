import cPickle
import pickle

from spacy.tokens.doc import Doc

from preprocessing import get_spacy_parser

"""
Serialize and deserialize spacy docs and associated metadata
"""


def serialize_docs(data, filepath):
    """Serialize a list of documents + associated metadata
    Args:
       data: list of dicts, "content" key has a spacy doc and other keys have metadata
       filepath: path to write 
    """
    data = data.copy()  # don't overwrite the spacy docs with binary data outside this scope
    for row in data:
        # turn doc objects into byte arrays
        row['content'] = row['content'].to_bytes()

    pickle.dump(data, open(filepath, 'wb'))


def read_docs(filepath):
    """Deserialize a list of documents + associated metadata"""
    spacy_parser = get_spacy_parser()
    data = pickle.load(open(filepath, 'rb'))
    for row in data:
        doc = Doc(spacy_parser.vocab)
        # read doc object from serialized byte array
        row['content'] = doc.from_bytes(row['content'])
    return data


def serialize_model(model, filepath):
    """Pickle an sklearn pipeline"""
    pickle.dump(model, open(filepath, 'wb'))


def read_model(filepath):
    """Load a pickled model"""
    return pickle.load(open(filepath, 'rb'))
