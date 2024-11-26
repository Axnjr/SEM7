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


# 3. Porter stemmer
is a widely used and influential stemming algorithm in Natural Language Processing (NLP). It was developed by Martin Porter in 1980 and is designed to reduce words to their root or base form (stem) by systematically removing suffixes. Unlike lemmatization, Porter stemming does not necessarily produce grammatically correct words but instead focuses on simplifying terms for information retrieval and text analysis.

## Steps of the Porter Stemming Algorithm:
The algorithm works by applying a series of rules to remove common suffixes from words. These rules are applied in a specific order to ensure that the stemming process is consistent and accurate. Here are the main steps:

- **Plurals and Past Participles:** Removes suffixes like 's', 'es', 'ed', 'ing' (e.g., "hopping" → "hop").
- **Derivational Suffixes:** Simplifies endings like 'ational' → 'ate' (e.g., "relational" → "relate").
- **Other Derivational Endings:** Strips suffixes like 'icate', 'ful', 'ness' (e.g., "hopefulness" → "hope").
- **Additional Suffixes:** Removes endings like 'al', 'ance', 'er' (e.g., "formalize" → "formal").
- **Final Adjustments:** Removes or retains 'e' based on its necessity (e.g., "probate" → "probat").

