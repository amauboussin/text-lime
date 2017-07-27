import itertools

import spacy

PARSING_BATCH_SIZE = 1000
PARSING_N_THREADS = -1

SPACY_PARSER = None


def get_spacy_parser():
    global SPACY_PARSER
    if SPACY_PARSER is None:
        SPACY_PARSER = spacy.load('en')
    return SPACY_PARSER


def parse_content_bulk(data):
    """Generator that takes a list of dicts parses the content field from unicode to a spacy doc"""
    # TODO(andrew): this isn't faster locally, spacy probably isn't installed with OpenMP
    # https://github.com/explosion/spaCy/issues/267
    spacy_parser = get_spacy_parser()

    #  copying example from https://github.com/explosion/spaCy/issues/172
    gen1, gen2 = itertools.tee(iter(data))
    text_iterator = (row['content'] for row in gen1)
    metadata_iterator = (row for row in gen2)

    doc_iterator = spacy_parser.pipe(text_iterator, batch_size=PARSING_BATCH_SIZE,
                                     n_threads=PARSING_N_THREADS)
    for doc, metadata in itertools.izip(doc_iterator, metadata_iterator):
        metadata['content'] = doc
        yield metadata


def parse_content_serial(data):
    """Parse the content field of a list of dicts from unicode to a spacy doc"""
    spacy_parser = get_spacy_parser()
    for row in data:
        row['content'] = spacy_parser(row['content'])
    return data
