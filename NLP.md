# `UNIT - 1 / 2`

# 1. Lexical Analysis (Tokenization): 
is the process of splitting text into smaller units, called tokens. Tokens can be words, sentences, or subwords, depending on the level of tokenization. 
## Examples
- Sentence Tokenization: Splitting a paragraph into sentences.
    - Input: `"I love NLP. It is fascinating!"`
    - Output: `[I love NLP.", "It is fascinating!"]`
- Word Tokenization: Splitting sentences into words or subwords.
    - Input: `"I love NLP!"`
    - Output: `["I", "love", "NLP", "!"]`

## Need of tokenization
- Effective Text Processing: Reduces the size of raw text for easier handling.
- Feature Extraction: Represents text data numerically for machine learning models.
- Language Modelling: Helps create organized representations of language.
- Information Retrieval: Essential for efficient indexing and searching.
- Text Analysis: Used in tasks like sentiment analysis and named entity recognition.
- Vocabulary Management: Manages a corpus’s vocabulary by generating distinct tokens.
- Task-Specific Adaptation: Customizable for specific NLP tasks.
- Preprocessing Step: Transforms unprocessed text for further analysis.


# 2. Stemming
is the process of reducing a word to its root form by removing prefixes or suffixes. The resulting stem might not always be a valid word, as the focus is on shortening rather than grammatical accuracy. **Example:**

- Input: `"running, runner, runs"`
- Output: `["run", "run", "run"]`
- Limitation:
Stemming may not always produce meaningful or grammatically correct words, e.g., `"better" → "bett"`.
A popular stemming algorithm is the Porter Stemmer.

## Benefits:

- Simplicity: 
Stemming is straightforward and computationally inexpensive.
- Speed: 
Faster processing time due to simple rules and lack of context consideration.
- Useful in applications where speed is crucial, such as search engines

## Limitations:

- Accuracy: 
Can produce stems that are not actual words, leading to less accurate results.
- Over-Stemming: 
Can sometimes strip too much off a word (e.g., “running” to “runn”).
- Under-Stemming: 
Can sometimes strip too little off a word (e.g., “running” to “run”).


