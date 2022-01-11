import unittest
from BPE import BytePairEncoding
import pickle


class TestBPEBasic(unittest.TestCase):
    def setUp(self) -> None:
        self.bpe = BytePairEncoding(
            corpus_path=r"tests\data\text\test_small.txt", lower_case=True
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


class TestBPECased(unittest.TestCase):
    def setUp(self) -> None:
        self.bpe = BytePairEncoding(
            corpus_path=r"tests\data\text\test_medium.txt", lower_case=False
        )

    def test_perform_BPE_cased(self):
        # Perform uncased BPE on small corpus
        self.bpe.create_vocab_and_tokenization(num_merges=250)

        with open(r"tests\data\vocab\bpe_expected_cased.pkl", "rb") as f:
            expected_vocab = pickle.load(f)

        self.assertEqual(set(expected_vocab), set(self.bpe.vocab))

    def test_tokenize_cased(self):
        self.bpe.load_tokenization(r"tests\data\tokenizations\tokenization_cased.json")

        string_to_tokenize = (
            "This is a test string! It contains UPPERCASE and lowercase"
        )
        expected_tokens = [
            273,
            101,
            63,
            10,
            11,
            29,
            84,
            29,
            57,
            80,
            295,
            37,
            13,
            84,
            0,
            48,
            109,
            54,
            45,
            25,
            19,
            19,
            27,
            41,
            295,
            295,
            295,
            27,
            37,
            66,
            135,
            17,
            59,
            62,
            29,
            47,
        ]

        tokens = self.bpe.tokenize(string_to_tokenize=string_to_tokenize)

        self.assertEqual(tokens.converted_tokens, expected_tokens)


class TestBPEUncased(unittest.TestCase):
    def setUp(self) -> None:
        self.bpe = BytePairEncoding(
            corpus_path=r"tests\data\text\test_medium.txt", lower_case=True
        )

    def test_perform_BPE_uncased(self):
        # Perform uncased BPE on small corpus
        self.bpe.create_vocab_and_tokenization(num_merges=250)

        with open(r"tests\data\vocab\bpe_expected_uncased.pkl", "rb") as f:
            expected_vocab = pickle.load(f)

        self.assertEqual(set(expected_vocab), set(self.bpe.vocab))

    def test_tokenize_uncased(self):
        self.bpe.load_tokenization(
            r"tests\data\tokenizations\tokenization_uncased.json"
        )

        string_to_tokenize = (
            "This is a test string! It contains UPPERCASE and lowercase"
        )
        expected_tokens = [
            146,
            89,
            52,
            17,
            22,
            1,
            56,
            1,
            45,
            69,
            283,
            25,
            32,
            56,
            27,
            36,
            98,
            41,
            33,
            10,
            0,
            0,
            47,
            51,
            1,
            35,
            55,
            121,
            5,
            47,
            51,
            1,
            35,
        ]

        tokens = self.bpe.tokenize(string_to_tokenize=string_to_tokenize)

        self.assertEqual(tokens.converted_tokens, expected_tokens)


if __name__ == "__main__":
    unittest.main()
