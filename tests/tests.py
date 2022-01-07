import unittest
from BPE import BytePairEncoding
import pickle


class TestBPE(unittest.TestCase):
    def setUp(self) -> None:
        self.bpe = BytePairEncoding(
            corpus_path=r"tests\test_small.txt", lower_case=True
        )

        self.bpe_full_uncased = BytePairEncoding(
            corpus_path=r"tests\test_medium.txt", lower_case=True
        )

        self.bpe_full_cased = BytePairEncoding(
            corpus_path=r"tests\test_medium.txt", lower_case=False
        )

    def test_split_into_words_and_create_vocab(self):
        vocab_expected = {
            "t h i s </w>": 1,
            "i s </w>": 1,
            "a </w>": 1,
            "t e s t </w>": 1,
            "s t r i n g </w>": 1,
            "! </w>": 1,
        }

        self.assertEqual(vocab_expected, self.bpe.split_into_words_and_create_vocab())

    def test_count_pairs(self):
        vocab = {
            "t h i s </w>": 1,
            "i s </w>": 1,
            "a </w>": 1,
            "t e s t </w>": 1,
            "s t r i n g </w>": 1,
            "! </w>": 1,
        }
        pairs_expected = {
            ("i", "s"): 2,
            ("s", "</w>"): 2,
            ("s", "t"): 2,
            ("t", "h"): 1,
            ("h", "i"): 1,
            ("a", "</w>"): 1,
            ("t", "e"): 1,
            ("e", "s"): 1,
            ("t", "</w>"): 1,
            ("t", "r"): 1,
            ("r", "i"): 1,
            ("i", "n"): 1,
            ("n", "g"): 1,
            ("g", "</w>"): 1,
            ("!", "</w>"): 1,
        }

        pairs_bpe = self.bpe.count_pairs(vocab)
        self.assertEqual(pairs_expected, pairs_bpe)

    def test_perform_merge(self):
        vocab = {
            "t h i s </w>": 1,
            "i s </w>": 1,
            "a </w>": 1,
            "t e s t </w>": 1,
            "s t r i n g </w>": 1,
            "! </w>": 1,
        }
        pairs_pattern = {
            ("i", "s"): 2,
            ("s", "</w>"): 2,
            ("s", "t"): 2,
            ("t", "h"): 1,
            ("h", "i"): 1,
            ("a", "</w>"): 1,
            ("t", "e"): 1,
            ("e", "s"): 1,
            ("t", "</w>"): 1,
            ("t", "r"): 1,
            ("r", "i"): 1,
            ("i", "n"): 1,
            ("n", "g"): 1,
            ("g", "</w>"): 1,
            ("!", "</w>"): 1,
        }

        merged_vocab_expected = {
            "t h is </w>": 1,
            "is </w>": 1,
            "a </w>": 1,
            "t e s t </w>": 1,
            "s t r i n g </w>": 1,
            "! </w>": 1,
        }

        merged_vocab, _ = self.bpe.perform_merge(vocab, pairs_pattern)

        self.assertEqual(merged_vocab_expected, merged_vocab)

    def test_perform_BPE_uncased(self):

        self.bpe_full_uncased.create_vocab_and_tokenization(num_merges=250)

        with open(r"tests\bpe_expected_uncased.pkl", "rb") as f:
            expected_vocab = pickle.load(f)

        self.assertEqual(set(expected_vocab), set(self.bpe_full_uncased.vocab))

    def test_perform_BPE_cased(self):

        self.bpe_full_cased.create_vocab_and_tokenization(num_merges=250)

        with open(r"tests\bpe_expected_cased.pkl", "rb") as f:
            expected_vocab = pickle.load(f)

        self.assertEqual(set(expected_vocab), set(self.bpe_full_cased.vocab))
