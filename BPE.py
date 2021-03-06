import collections, re
from typing import Iterable, NamedTuple
import nltk
import json
from tqdm import tqdm


class Tokenization(NamedTuple):
    tokens: Iterable[str]
    converted_tokens: Iterable[int]


class BytePairEncoding:
    """
    Class to perform Byte-Pair Encoding (BPE) on a givin corpus.

    As outlined in 'Neural Machine Translation of Rare Words with Subword Units'
    <https://arxiv.org/abs/1508.07909>

    """

    def __init__(
        self,
        corpus_path,
        lower_case,
        EOW_TOKEN="</w>",
        UNK_TOKEN="<UNK>",
        PAD_TOKEN="<PAD>",
    ):
        with open(corpus_path, encoding="utf-8") as f:
            self.corpus = f.read()

        if lower_case:
            self.corpus = self.corpus.lower()

        self.lower_case = lower_case
        print(
            f"Base corpus has {len(self.corpus)} characters with {len(set(self.corpus))} distinct."
        )

        self.EOW_TOKEN = EOW_TOKEN
        self.UNK_TOKEN = UNK_TOKEN
        self.PAD_TOKEN = PAD_TOKEN

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
    def count_pairs(self, vocab):
        """
        From a vocabulary, count all pairs of symbols and return a sorted dict with all pair counts

        Ex:
        >>> vocab = {'t h i s </w>': 1, 'i s </w>': 1, 'a </w>': 1, 't e s t </w>': 1, 's t r i n g </w>': 1, '! </w>': 1}
        >>> print(count_pairs(vocab))
        {('i', 's'): 2, ('s', '</w>'): 2, ('s', 't'): 2, ('t', 'h'): 1, ('h', 'i'): 1, ('a', '</w>'): 1, ('t', 'e'): 1, ('e', 's'): 1, ('t',
        '</w>'): 1, ('t', 'r'): 1, ('r', 'i'): 1, ('i', 'n'): 1, ('n', 'g'): 1, ('g', '</w>'): 1, ('!', '</w>'): 1}
        """

        pairs = collections.defaultdict(int)
        pairs_pattern = collections.defaultdict(int)

        for word, freq in vocab.items():

            # Iterate through each word and count pairs
            for idx in range(len(word.split()) - 1):
                pairs["".join(word.split()[idx] + word.split()[idx + 1])] += freq
                pairs_pattern[(word.split()[idx], word.split()[idx + 1])] += freq

        return dict(sorted(pairs_pattern.items(), key=lambda x: x[1], reverse=True))

    def perform_merge(self, vocab, pairs_pattern):
        """
        From a (possibly merged) vocabulary, perform a merge on the given pattern

        Ex:
        >>> vocab = {'t h i s </w>': 1, 'i s </w>': 1, 'a </w>': 1, 't e s t </w>': 1, 's t r i n g </w>': 1, '! </w>': 1}
        >>> pairs_pattern = {('i', 's'): 2, ('s', '</w>'): 2, ('s', 't'): 2, ('t', 'h'): 1, ('h', 'i'): 1, ('a', '</w>'): 1, ('t', 'e'): 1, ('e', 's'): 1, ('t', '</w>'): 1, ('t', 'r'): 1, ('r', 'i'): 1, ('i', 'n'): 1, ('n', 'g'): 1, ('g', '</w>'): 1, ('!', '</w>'): 1}
        >>> print(perform_merge(vocab, pairs_pattern))
        {'t h is </w>': 1, 'is </w>': 1, 'a </w>': 1, 't e s t </w>': 1, 's t r i n g </w>': 1, '! </w>': 1}
        """

        if len(pairs_pattern.keys()) == 0:
            return vocab, None

        else:
            pattern_find = list(pairs_pattern.keys())[0]

            pattern_A = pattern_find[0]
            pattern_B = pattern_find[1]

            bigram = " ".join([pattern_A, pattern_B])

            pattern = "".join([pattern_A, pattern_B])

            merged_vocab = collections.defaultdict(int)

            # This regex case here is weird... had to look at the paper for the negative lookbehinds.
            # Without this, for example, the merge rule ('e','l') would produce 'el','del','rel'
            regex_pattern = re.compile(r"(?<!\S)" + re.escape(bigram) + r"(?!\S)")

            for word, freq in vocab.items():

                repl = re.sub(regex_pattern, pattern, word)
                merged_vocab[repl] += freq

            return merged_vocab, pattern

    def perform_BPE(self, num_merges):

        vocab = self.split_into_words_and_create_vocab()
        # Create the base vocab of all symbols and progressively add to it
        self.vocab = self.create_vocab(vocab)

        for i in tqdm(range(num_merges)):
            pairs_pattern = self.count_pairs(vocab)
            vocab, pattern = self.perform_merge(vocab, pairs_pattern)

            if pattern is not None:
                self.vocab.append(pattern)

        self.vocab += [self.UNK_TOKEN, self.PAD_TOKEN]
        return vocab

    def create_vocab(self, bpe_vocab):
        vocab = []

        for word, _ in bpe_vocab.items():
            for token in word.split():
                vocab.append(token)

        return list(set(vocab))

    def create_tokenization(self, vocab, save_tokenization=True):

        itos = {i: vocab[i] for i in range(len(vocab))}

        stoi = {vocab[i]: i for i in range(len(vocab))}

        if save_tokenization:
            with open("saved_tokenizations/tokenization.json", "w") as j:
                json.dump(itos, j, indent=4)
        return stoi, itos

    def load_tokenization(self, tokenization_path):

        with open(tokenization_path, "r") as f:
            saved_dict = json.load(f)

        self.itos = {int(k): v for k, v in saved_dict.items()}

        self.stoi = {v: int(k) for k, v in saved_dict.items()}

        self.vocab = list(saved_dict.values())

    def create_vocab_and_tokenization(self, num_merges):
        BPE_vocab = self.perform_BPE(num_merges=num_merges)

        tokens_stoi, tokens_itos = self.create_tokenization(self.vocab)

        self.stoi = tokens_stoi
        self.itos = tokens_itos

    def tokenize(self, string_to_tokenize):

        assert (
            self.stoi is not None and self.itos is not None
        ), "Requires tokenization of base corpus first."

        # Split string into characters and apply merge rules

        split_chars = []
        for word in nltk.wordpunct_tokenize(
            string_to_tokenize.lower() if self.lower_case else string_to_tokenize
        ):
            split_chars.append(" ".join(list(word) + [self.EOW_TOKEN]))

        word_tokenization = []

        for word in tqdm(split_chars):
            for token in sorted(self.vocab, key=len, reverse=True):
                # Splits BPE tokens like 'mathbb</w>' -> ['mathbb', '</w>', '']
                split_tok = re.split(f"({self.EOW_TOKEN})", token)
                if len(split_tok) > 1:
                    pattern = " ".join(list(split_tok[0]) + [split_tok[1]])
                else:
                    pattern = " ".join(list(split_tok[0]))

                # ugh.
                if pattern == "\\":
                    word = re.sub(
                        r"(?<!\S)" + re.escape(pattern) + r"(?!\S)",
                        repl=re.escape(token),
                        string=word,
                    )
                else:
                    word = re.sub(
                        r"(?<!\S)" + re.escape(pattern) + r"(?!\S)",
                        repl=token,
                        string=word,
                    )
            word_tokenization += word.split()

        # Replace unknown tokens with UNK TOKEN
        word_tokenization = [
            i if i in self.stoi.keys() else self.UNK_TOKEN for i in word_tokenization
        ]
        
        return Tokenization(word_tokenization, [self.stoi[i] for i in word_tokenization] )

    def tokens_to_str(self, tokens):
        """
        Takes a list of tokens and converts it back to text.
        """

        concat_str = "".join([self.itos[i] for i in tokens])

        return " ".join(concat_str.split(self.EOW_TOKEN))


if __name__ == "__main__":

    bpe = BytePairEncoding(corpus_path=r"data\corpus.txt", lower_case=False)

    # bpe.create_vocab_and_tokenization(num_merges=250)

    bpe.load_tokenization(r"saved_tokenizations\tokenization.json")

    string_to_tokenize = """
    Byte pair encoding[1][2] or digram coding[3] is a simple form of data compression in which the most common pair of consecutive bytes of data is
    replaced with a byte that does not occur within that data. A table of the replacements is required to rebuild the original data.
    The algorithm was first described publicly by Philip Gage in a February 1994 article "A New Algorithm for Data Compression" in the C Users Journal.
    [4]

    A variant of the technique has shown to be useful in several natural language processing (NLP) applications, such as Google's SentencePiece,[5]
    and OpenAI's GPT-3.[6]
    """

    tokens = bpe.tokenize(string_to_tokenize=string_to_tokenize)
    # print(bpe.tokens_to_str(tokens))

    print(tokens.tokens)
    print(tokens.converted_tokens)
