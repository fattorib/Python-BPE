import unittest
from BPE import BytePairEncoding

class TestBPE(unittest.TestCase):
    
    def setUp(self) -> None:
        self.bpe = BytePairEncoding(corpus_path=r'tests\test_small.txt', lower_case=True)
    
    def test_split_into_words_and_create_vocab(self):
        vocab = {'t h i s </w>': 1, 'i s </w>': 1, 'a </w>': 1, 't e s t </w>': 1, 's t r i n g </w>': 1, '! </w>': 1}

        self.assertEqual(vocab, self.bpe.split_into_words_and_create_vocab())

    def test_count_pairs(self):
        vocab = {'t h i s </w>': 1, 'i s </w>': 1, 'a </w>': 1, 't e s t </w>': 1, 's t r i n g </w>': 1, '! </w>': 1}
        pairs = {'is': 2, 's</w>': 2, 'st': 2, 'th': 1, 'hi': 1, 'a</w>': 1, 'te': 1, 'es': 1, 't</w>': 1, 'tr': 1, 'ri': 1, 'in': 1, 'ng': 1, 'g</w>': 1, '!</w>': 1}

        pairs_bpe, _ =  self.bpe.count_pairs(vocab)
        self.assertEqual(pairs, pairs_bpe)

    def perform_merge(self):
        pass