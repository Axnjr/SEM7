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

# 4. Segmentation
It is similar to `tokenization` but we focus on spiltting the sentences or paragraphs by `logical boundries` rather than just words or sentences. 

For example, the paragraph "I love programming. It's both challenging and rewarding." can be segmented into:
- "I love programming."
- "It's both challenging and rewarding."

# 5. Lemmatization
Lemmatization reduces words to their `dictionary base form` (lemma) while `preserving grammatical meaning`. Unlike stemming, it ensures that the output is a valid word. **Example:** for input 
- `better → good`, `caring → care`
- in case of `stemming`: `caring → car`

## How Lemmatization Works? Lemmatization involves several steps:

- **Part-of-Speech (POS) Tagging:** 
Identifying the grammatical category of each word (e.g., noun, verb, adjective).

- **Morphological Analysis:** 
Analyzing the structure of the word to understand its root form.

- **Dictionary Lookup:** 
Using a predefined vocabulary to find the lemma of the word. For example, the word “better” would be lemmatized to “good” if it is identified as an adjective, whereas “running” would be lemmatized to “run” if identified as a verb.

## Techniques in Lemmatization

- **Rule-Based Lemmatization:** 
Uses predefined grammatical rules to transform words. For instance, removing the “-ed” suffix from regular past tense verbs.

- **Dictionary-Based Lemmatization:** 
Looks up words in a dictionary to find their base forms.

- **Machine Learning-Based Lemmatization:** 
Employs machine learning models trained on annotated corpora to predict the lemma of a word.

## Benefits:
- **Accuracy:** 
Lemmatization provides more accurate results because it considers the context and meaning of words.
- **Standardization:** 
Ensures words are reduced to their dictionary form, aiding in tasks like text normalization and information retrieval.

## Limitations:

- **Complexity:** 
Requires more computational resources and a comprehensive dictionary.
- **Dependency on POS Tagging:** 
Requires accurate POS tagging, which adds to the processing overhead.


# 6. The Edit Distance algorithm
also known as `Levenshtein Distance`, is a way to measure how `different` two strings are by calculating the `minimum number of operations` required to `transform` one string into another. It is widely used in NLP for tasks like `spelling correction, approximate string matching, and fuzzy search`.

## The algorithm allows three basic operations:
- **Insertion:** Adding a character.
E.g., Transform "cat" → "cats" (insert 's').
- **Deletion:** Removing a character.
E.g., Transform "cats" → "cat" (delete 's').
- **Substitution:** Replacing one character with another.
E.g., Transform "cat" → "cut" (substitute 'a' with 'u').

Each operation is assumed to have a cost of 1 (though this can vary in some implementations). 

## Working
The algorithm use a `dynamic programming` based approch to calculate the `edit distance` efficiently. It uses a `matrix` to compute distances in a `step-by-step manner`.

- **Matrix Setup:**
Create a table (matrix) where `rows represent the characters of one string` and `columns represent the other`. The table tracks the minimum number of operations required to align substrings.

- **Initialization:**
    - Fill the first row with incremental costs for `transforming an empty string into the second string` (insertions).

    - Fill the first column with incremental costs for `transforming the first string into an empty string` (deletions).

- **Dynamic Programming Update:** For each cell (i, j) in the matrix: Calculate the minimum cost using formula:

`dp[i][j] = min(dp[i][j - 1] + 1, dp[i - 1][j] + 1, dp[i - 1][j - 1] + cost)`

The value in the `bottom-right cell` of the matrix gives the `total minimum edit distance between the two strings`.

## Applications in NLP
- **Spelling Correction:**
Suggest words with a small edit distance from the misspelled word. E.g., "recieve" → "receive" (edit distance = 1).
- **Plagiarism Detection:**
Measure similarity between texts by computing edit distances.
- **DNA Sequence Alignment:**
Compare biological sequences.
- **Autocomplete Systems:**
Rank suggestions based on their edit distances from the input query.


# 7. Collocations: 
are `word pairs` or groups of words that `frequently appear together` in a language, forming a `natural combination`. These combinations are used by native speakers to express ideas more naturally. Understanding collocations is important in Natural Language Processing (NLP) as they often carry specific meanings or contexts that differ from their individual words.  Examples: `strong tea, heavy rain, big mistake`. "Strong tea" sounds natural, while "powerful tea" does not.


# 8. Morphological Analysis: 
in Natural Language Processing (NLP) is the process of `analyzing the structure of words` to understand `their components and relationships`.

Morphology is the branch of `linguistics` concerned with the `structure and form of words in a language`. Morphological analysis, in the context of NLP, refers to the `computational processing of word structures`. It aims to `break down` words into their `constituent parts`, such as roots, prefixes, and suffixes, `and understand their roles and meanings`. This process is essential for various NLP tasks, including language modeling, text analysis, and machine translation.

## Importance of Morphological Analysis
- **Understanding Word Formation:** It helps in identifying the basic building blocks of words, which is crucial for language comprehension.
- **Improving Text Analysis:** By breaking down words into their roots and affixes, it enhances the accuracy of text analysis tasks like sentiment analysis and topic modeling.
- **Enhancing Language Models:** Morphological analysis provides detailed insights into word formation, improving the performance of language models used in tasks like speech recognition and text generation.
- **Facilitating Multilingual - Processing:** It aids in handling the morphological diversity of different languages, making NLP systems more robust and versatile.

## Key techniques used in morphological analysis:

- **Stemming**
- **Lemmetization**
- **Morphological Parsing:**
involves analyzing the structure of words to identify their morphemes (roots, prefixes, suffixes). It requires knowledge of morphological rules and patterns. FSTs are computational models used to represent and analyze the morphological structure of words. They consist of states and transitions, capturing the rules of word formation.
- **Neural network models:** 
especially deep learning models, can be trained to perform morphological analysis by learning patterns from large datasets.
- **Rule-based methods:** 
rely on manually defined linguistic rules for morphological analysis. These rules can handle specific language patterns and exceptions.
    - Applications:
        - Affix Stripping: Removing known prefixes and suffixes to find the root form.
        - Inflectional Analysis: Identifying grammatical variations like tense, number, and case.

## Morphology Types:

- **Derivational Morphology:** 
The process of creating new words by adding prefixes or suffixes to a root, often changing the word's meaning or part of speech.
Example: happy → happiness (adjective → noun).
- **Inflectional Morphology:**
The process of adding grammatical information to a word (e.g., tense, number, gender) without changing its core meaning or part of speech.
Example: walk → walked (verb → past tense verb).

