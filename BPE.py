from typing import final
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

    def __init__(self, corpus_path, lower_case, EOW_TOKEN = '</w>', UNK_TOKEN = '_UNK_'):
        with open(corpus_path, encoding='utf-8') as f:
            self.corpus = f.read()

        if lower_case:
            self.corpus = self.corpus.lower()

        self.lower_case = lower_case
        print(f'Base corpus has {len(self.corpus)} characters with {len(set(self.corpus))} distinct.')

        self.EOW_TOKEN = EOW_TOKEN
        self.UNK_TOKEN = UNK_TOKEN
       

    # ---------- Preprocessing ---------- #
    def split_into_words_and_create_vocab(self):
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
            vocab[" ".join(list(word) + [self.EOW_TOKEN])] += 1

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

        pattern_A = pattern_find[0]
        pattern_B = pattern_find[1]

        bigram = " ".join([pattern_A, pattern_B])

        # print(f'Best token: {pattern_find} -> {pattern}')
        
        merged_vocab = collections.defaultdict(int)

        for word, freq in vocab.items():

            #This regex case here is weird... had to look at the paper for the negative lookbehinds.
            # Without this, for example, the merge rule ('e','l') would produce 'el','del','rel'
            repl = re.sub(r'(?<!\S)' + re.escape(bigram) + r'(?!\S)', pattern, word)
            merged_vocab[repl] += freq

        return merged_vocab, pattern
       
    def perform_BPE(self,num_merges):

        vocab = self.split_into_words_and_create_vocab()
        #Create the base vocab of all symbols and progressively add to it 
        self.vocab = self.create_vocab(vocab)

        for i in tqdm(range(num_merges)):
            pairs, pairs_pattern = self.count_pairs(vocab)
            vocab, pattern = self.perform_merge(vocab, pairs, pairs_pattern)
            if pattern is not None:
                self.vocab.append(pattern)
            # print(f'Number of tokens at {i}th merge {len(self.vocab)}')

        return vocab

    def create_vocab(self,bpe_vocab):
        vocab = []

        for word, _ in bpe_vocab.items():
            for token in word.split():
                vocab.append(token)

        # return sorted(list(set(vocab)))
        return list(set(vocab))

    def create_tokenization(self,vocab, save_pkl = True):
        
        itos = {i:vocab[i] for i in range(len(vocab))}

        stoi = {vocab[i]:i for i in range(len(vocab))}

        # if save_pkl:
        #     pickle.dump(stoi,open(f'BPE/stoi_vocab_{len(chars)}.pkl', 'wb'))
        #     pickle.dump(itos,open(f'BPE/itos_vocab_{len(chars)}.pkl', 'wb'))
        return stoi, itos

    def create_vocab_and_tokenization(self,num_merges):
        BPE_vocab = self.perform_BPE(num_merges=num_merges)

    
        # final_vocab = self.create_vocab(bpe_vocab=BPE_vocab)

        tokens_stoi, tokens_itos = self.create_tokenization(self.vocab)

        self.stoi = tokens_stoi
        self.itos = tokens_itos

    
    def tokenize(self,string_to_tokenize):

        assert self.stoi is not None and self.itos is not None, 'Requires tokenization of base corpus first.'

        #Split string into characters and apply merge rules

        split_chars = []
        for word in nltk.wordpunct_tokenize(string_to_tokenize.lower() if self.lower_case else string_to_tokenize):
            split_chars.append(" ".join(list(word) + [self.EOW_TOKEN]))

       
        word_tokenization = []
    
        for word in split_chars:
            # for token in sorted(self.vocab, key = len, reverse=True):    
            for token in self.vocab:        
                #Splits BPE tokens like 'mathbb</w>' -> ['mathbb', '</w>', '']
                split_tok = re.split('(</w>)', token)
                if len(split_tok) > 1:
                    pattern = " ".join(list(split_tok[0])+[split_tok[1]])
                else:
                    pattern = " ".join(list(split_tok[0]))
                
                #ugh.
                if pattern == '\\':
                    word = re.sub(r'(?<!\S)' + re.escape(pattern) + r'(?!\S)', repl=re.escape(token), string = word)
                else:
                    word = re.sub(r'(?<!\S)' + re.escape(pattern) + r'(?!\S)', repl=token, string = word)
        
            word_tokenization += word.split()
        print(word_tokenization)
        return [self.stoi[i] for i in word_tokenization]




if __name__ == "__main__":

    BPE = BytePairEncoding(corpus_path= 'corpus.txt', lower_case = True)

    BPE.create_vocab_and_tokenization(num_merges=50)

    print(BPE.tokenize(string_to_tokenize='This is a test sentence we are trying to tokenize. Lets see what happens. manifold Frobenius! Harry!'))


