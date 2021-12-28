import numpy as np
import collections, re
import nltk


def split_into_words(string, EOW_TOKEN="</w>"):
    """
    Split text into list of words and add EOW token to each.
    Create vocab of all words with their counts.

    Ex:
    >>> print(split_into_words_string('this is a test string!'))
    {'t h i s </w>': 1, 'i s </w>': 1, 'a </w>': 1, 't e s t </w>': 1, 's t r i n g </w>': 1, '! </w>': 1}

    """

    split_string = nltk.wordpunct_tokenize(string)

    vocab = collections.defaultdict(int)
    for word in split_string:
        vocab[" ".join(list(word) + [EOW_TOKEN])] += 1

    return vocab


def count_pairs(vocab):
    """
    From a vocabulary, count all pairs of symbols and return a sorted dict with all pair counts

    Ex:
    >>> vocab = {'t h i s </w>': 1, 'i s </w>': 1, 'a </w>': 1, 't e s t </w>': 1, 's t r i n g </w>': 1, '! </w>': 1}
    >>> print(count_pairs(vocab))
    {'is': 2, 's</w>': 2, 'st': 2, 'th': 1, 'hi': 1, 'a</w>': 1, 'te': 1, 'es': 1, 't</w>': 1, 'tr': 1, 'ri': 1, 'in': 1, 'ng': 1, 'g</w>': 1, '!</w>': 1}
    """

    pairs = collections.defaultdict(int)
    pairs_pattern = collections.defaultdict(int)

    for word, freq in vocab.items():

        # Iterate through each word and count pairs
        for idx in range(len(word.split()) - 1):
            pairs["".join(word.split()[idx] + word.split()[idx + 1])] += freq
            pairs_pattern[(word.split()[idx],word.split()[idx + 1])] += freq

    return dict(sorted(pairs.items(),  key=lambda x: x[1], reverse=True)), dict(sorted(pairs_pattern.items(),  key=lambda x: x[1], reverse=True))

def perform_merge(vocab, pairs, pairs_pattern):
    """
    From a (possibly merged) vocabulary, perform a merge on the given pattern

    Ex:
    >>> vocab = {'t h i s </w>': 1, 'i s </w>': 1, 'a </w>': 1, 't e s t </w>': 1, 's t r i n g </w>': 1, '! </w>': 1}
    >>> pairs = {'is': 2, 's</w>': 2, 'st': 2, 'th': 1, 'hi': 1, 'a</w>': 1, 'te': 1, 'es': 1, 't</w>': 1, 'tr': 1, 'ri': 1, 'in': 1, 'ng': 1, 'g</w>': 1, '!</w>': 1}
    >>> pairs_pattern = {('i', 's'): 2, ('s', '</w>'): 2, ('s', 't'): 2, ('t', 'h'): 1, ('h', 'i'): 1, ('a', '</w>'): 1, ('t', 'e'): 1, ('e', 's'): 1, ('t', '</w>'): 1, ('t', 'r'): 1, ('r', 'i'): 1, ('i', 'n'): 1, ('n', 'g'): 1, ('g', '</w>'): 1, ('!', '</w>'): 1}
    >>> print(perform_merge(vocab, pairs, pairs_pattern))
    {'t h is </w>': 1, 'is </w>': 1, 'a </w>': 1, 't e s t </w>': 1, 's t r i n g </w>': 1, '! </w>': 1}
    """

    pattern = list(pairs.keys())[0]

    pattern_find = list(pairs_pattern.keys())[0]

    bigram = pattern_find[0] + ' ' + pattern_find[1]

    merged_vocab = collections.defaultdict(int)

    for word, freq in vocab.items():
        repl = re.sub(bigram, pattern, word)
        merged_vocab[repl] += freq

    return merged_vocab
       
def perform_BPE(base_corpus, num_merges):

    vocab = split_into_words(base_corpus)

    for i in range(num_merges):

        pairs, pairs_pattern = count_pairs(vocab)
        vocab = perform_merge(vocab, pairs, pairs_pattern)

    return vocab

if __name__ == "__main__":

    string = "this is a test string!"

    # print(perform_merge(vocab, pairs))

    print(perform_BPE(base_corpus=string, num_merges=5))






