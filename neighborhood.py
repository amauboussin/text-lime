import itertools

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances

NEIGHBORS_TO_CONSIDER = 100
#  don't replace puntuanction, pronouns, and other non-semantic tags
#  https://spacy.io/docs/usage/pos-tagging
TAG_BLACKLIST = {'PUNCT', '-LRB-', '-RRB-', ',', ':', '.',
                 '``', "''", '""', '#', '$', 'HVS', 'HYPH', 'PRP', 'PRP$', 'SP',
                 'ADD', 'DT', 'IN', 'EX', 'XX'}
similarity_function = euclidean_distances
bigger_is_closer = False


def softmax(x, temp):
    """Returns softmax probabilities with temperature"""
    e_x = np.exp(x / temp)
    return e_x / e_x.sum()


def get_neighboring_docs(dataset, doc, max_subset_size, softmax_temp=1.):

    token_list = [token.text.lower() for token in doc]
    masks = get_all_masks(len(token_list), max_subset_size)

    #  sample replacement tokens
    n_samples = len(masks)
    replacement_tokens = []
    for i, token in enumerate(doc):
        token_text = token.text.lower()
        alllow_token_to_be_replaced = True
        if token.tag_ in TAG_BLACKLIST or dataset.ft_vocab[token.tag_].size < NEIGHBORS_TO_CONSIDER:
            alllow_token_to_be_replaced = False
        else:
            no_word_vector = len(dataset.get_word_vector(token_text)) == 0
            empty_tag = len(dataset.ft_matrices[token.tag_]) == 0

            # broke the "can token be replaced" check into two conditions
            # to avoid possible key errors
            if no_word_vector or empty_tag:
                alllow_token_to_be_replaced = False

        if alllow_token_to_be_replaced:
            replacement_tokens.append(choose_new_tokens(token_text,
                                                        dataset.get_word_vector(token_text),
                                                        dataset.ft_matrices[token.tag_],
                                                        dataset.ft_vocab[token.tag_],
                                                        n_samples, softmax_temp))
        else:  # always replace with the same token
            replacement_tokens.append([(token_text, 0.)] * n_samples)

    #  create replacement token list for each doc
    neighborhood_tokens = []
    neighborhood_perturbations = []
    for mask_number, mask in enumerate(masks):

        new_tokens = [replacement_tokens[i][mask_number][0] if value else token_list[i]
                      for i, value in enumerate(mask)]

        local_regression_example = [replacement_tokens[i][mask_number][1] if value else 0.
                                    for i, value in enumerate(mask)]

        neighborhood_tokens.append(new_tokens)
        neighborhood_perturbations.append(local_regression_example)

    return np.array(neighborhood_tokens), np.array(neighborhood_perturbations)


def choose_new_tokens(token, token_vector, tag_matrix, tag_vocab,
                      n_samples, softmax_temp):
    """Sample replacement tokens"""
    # remove token itself from consideration
    mask = (tag_vocab != token)
    tag_vocab = tag_vocab[mask]
    tag_matrix = tag_matrix[mask]

    all_similarities = similarity_function(tag_matrix, token_vector.reshape(1, -1)).squeeze()
    all_similarities = -all_similarities if bigger_is_closer else all_similarities

    most_similar_indices = all_similarities.argsort()[:NEIGHBORS_TO_CONSIDER]
    similar_tokens = tag_vocab[most_similar_indices]
    token_distances = all_similarities[most_similar_indices]

    # normalize distances
    token_distances = (token_distances - token_distances.mean()) - token_distances.std()

    #  put negative distances into the softmax so smaller magnitudes are sampled more frequently
    token_probabilities = softmax(-all_similarities[most_similar_indices], softmax_temp)
    selected_indices = np.random.choice(np.arange(token_distances.size),
                                        size=n_samples, p=token_probabilities)

    return [(similar_tokens[i], token_distances[i]) for i in selected_indices]



def get_n_gram_masks(n_tokens, max_ngram_size):
    """Get a mask for every n_gram, from unigrams to max_ngram_size"""
    masks = []
    for i in range(n_tokens):
        tokens_until_end = n_tokens - i
        for n in range(1, min(max_ngram_size, tokens_until_end) + 1):
            n_gram_mask = np.zeros(n_tokens)
            n_gram_mask[i:i + n] = 1
            masks.append(n_gram_mask)
    return np.array(masks)


def get_all_masks(n_tokens, max_subset_size):
    """Get a mask for every token subset, for all sets size from 1 to max_ngram_size"""
    masks = []
    for subset_size in range(1, max_subset_size + 1):
        index_combos = itertools.combinations(range(n_tokens), subset_size)
        for indices_to_allow in index_combos:
            mask = np.zeros(n_tokens)
            mask[np.array(indices_to_allow)] = 1
            masks.append(mask)
    return np.array(masks)



# sample n_masks words for each position
# iterate through masks