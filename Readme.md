# Python Implementation of BPE 
Pedagogical implementation of Byte-Pair Encoding in pure Python. 

## Example Use

```python
#initialize BPE. Need to specify corpus path and whether to use a cased vocabulary
bpe = BytePairEncoding(corpus_path="your_corpus_here.txt", lower_case=False)

#perform merges
bpe.create_vocab_and_tokenization(num_merges=5000)

string_to_tokenize = """
Byte pair encoding[1][2] or digram coding[3] is a simple form of data compression in which the most common pair of consecutive bytes of data is
replaced with a byte that does not occur within that data. A table of the replacements is required to rebuild the original data.
The algorithm was first described publicly by Philip Gage in a February 1994 article "A New Algorithm for Data Compression" in the C Users Journal.
[4]

A variant of the technique has shown to be useful in several natural language processing (NLP) applications, such as Google's SentencePiece,[5]
and OpenAI's GPT-3.[6]
"""

#tokenize string
tokens = bpe.tokenize(string_to_tokenize=string_to_tokenize)
```

## Testing
```
python -m pytest
```

## References
BPE Paper: [https://arxiv.org/abs/1508.07909](https://arxiv.org/abs/1508.07909)
