import numpy as np
import collections, re
import nltk
import pickle
from tqdm import tqdm

class BytePairEncoding():
    """
    Class to perform Byte-Pair Encoding (BPE) on a givin corpus. 

    As outlined in 'Neural Machine Translation of Rare Words with Subword Units'
    <https://arxiv.org/abs/1508.07909>
    
    """

    def __init__(self, corpus_path):
        with open(corpus_path) as f:
            self.corpus = f.read()

        print(f'Base corpus has {len(self.corpus)} characters with {len(set(self.corpus))} distinct.')

    # ---------- Preprocessing ---------- #
    def split_into_words(self, EOW_TOKEN="</w>"):
        """
        Split text into list of words and add EOW token to each.
        Create vocab of all words with their counts.

        Ex:
        >>> print(split_into_words_string('this is a test string!'))
        {'t h i s </w>': 1, 'i s </w>': 1, 'a </w>': 1, 't e s t </w>': 1, 's t r i n g </w>': 1, '! </w>': 1}

        """

        split_string = nltk.wordpunct_tokenize(self.corpus)

        vocab = collections.defaultdict(int)
        for word in split_string:
            vocab[" ".join(list(word) + [EOW_TOKEN])] += 1

        return vocab

    # ---------- BPE Helper Functions ---------- #
    def count_pairs(self,vocab):
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

    def perform_merge(self,vocab, pairs, pairs_pattern):
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

        # print(pattern, pattern_find)

        #Need to remap special characters

        SPECIAL_CHARS = ['^','$','|','?','*','+','(',')','[',']','{','}','.']

        pattern_A = pattern_find[0]
        pattern_B = pattern_find[1]

        pattern_A = re.sub(r'\\', r'\\\\', pattern_A)
        pattern_B = re.sub(r'\\', r'\\\\', pattern_B)
        for c in SPECIAL_CHARS:
            pattern_A = re.sub('\\'+c, '\\'+c, pattern_A)
            pattern_B = re.sub('\\'+c, '\\'+c, pattern_B)

        bigram = pattern_A + ' ' + pattern_B
        
        merged_vocab = collections.defaultdict(int)

        for word, freq in vocab.items():
            repl = re.sub(bigram, pattern, word)
            merged_vocab[repl] += freq

        return merged_vocab
       
    def perform_BPE(self,num_merges):

        vocab = self.split_into_words()

        for i in tqdm(range(num_merges)):

            pairs, pairs_pattern = self.count_pairs(vocab)
            vocab = self.perform_merge(vocab, pairs, pairs_pattern)

        return vocab

    def create_final_vocab(self,bpe_vocab):
        vocab = []

        for word, _ in bpe_vocab.items():
            for token in word.split():
                vocab.append(token)

        return sorted(list(set(vocab)))

    def create_tokenization(self,vocab, save_pkl = True):
        
        itos = {i:vocab[i] for i in range(len(vocab))}

        stoi = {vocab[i]:i for i in range(len(vocab))}

        # if save_pkl:
        #     pickle.dump(stoi,open(f'BPE/stoi_vocab_{len(chars)}.pkl', 'wb'))
        #     pickle.dump(itos,open(f'BPE/itos_vocab_{len(chars)}.pkl', 'wb'))
        return stoi, itos

    def create_vocab_and_tokenization(self,num_merges):
        BPE_vocab = self.perform_BPE(num_merges=num_merges)

        final_vocab = self.create_final_vocab(bpe_vocab=BPE_vocab)

        print(final_vocab)

        tokens_stoi, tokens_itos = self.create_tokenization(final_vocab)

if __name__ == "__main__":

    # string = "this is a test string!"

    BPE = BytePairEncoding(corpus_path= 'corpus.txt')

    BPE.create_vocab_and_tokenization(num_merges=100)


