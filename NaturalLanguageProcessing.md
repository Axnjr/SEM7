// TO BE DONE
3. Explain Porter’s Stemming algorithm with  example.
4. Explain with suitable example following relationships between word meanings: Homonymy, Polysemy, Synonymy,Hyponomy,Hypernomy,Meronomy,Antonomy.
11. Explain edit distance algorithm with an example. Show working of the minimum no of operations required to transform “kitten” into  “sitting”.

// DONE

2. Discuss various challenges in processing natural languages.

10. What is POS tagging? List different approaches of POS tagging. Explain anyone in brief.

13. Write a short note on Wordnet.

14. What is the difference between stemming and lemmitization?

15. Write a short note on Training and Testing of dataset.

5. What is Natural language processing (NLP)? Discuss various stages involved in NLP process with suitable example.

8. What is Natural language processing? Explain ambiguity in Natural languages with suitable examples.

6. Explain N-gram model with example.

18. Define affixes . Explain the types of affixes.

20. Differentiate between syntactic ambiguity and lexical ambiguity.

// NOT DONE
1. Compare derivational and inflectional morphology with suitable example.
17. Describe open and closed class words in english with examples.
21. Explain the role of FSA in morphological analysis.

// TO BE DONE / WATCHED
12. Consider the following corpus:
<s> she asks you to wait patiently</s>
<s> he wants me to help him</s>
<s> they expect us to arrive early</s>
List all possible bigrams . Compute conditional probabilities and perdict the next word for the word “to”.

19. Consider following corpus
<s>I tell to sleep and rest</s>
<s> I would like to sleep for an hour</s>
<s> Sleep helps one to relax</s>
List all possible bigrams. Compute conditional probability and predict the next word for the word “to”.

7. Consider following Training data:
I am Sam 
Sam I am 
Sam I like 
Sam I do like
do I like Sam 
Assume that we use a bigram language model based on the above training data. What is the most probable next word predicted by the model for the following word sequences? 
(1) Sam ...
(2) Sam I do ... 
(3) Sam I am Sam ...
(4) do I like



# 1. Natural Language Processing: 
is a field within artificial intelligence that focuses on the interaction between computers and human language. It involves enabling machines to understand, interpret, and generate human language in a way that is both meaningful and useful.

> Stages of NLP

- Lexical Analysis (Tokenization): 
This stage involves breaking down a text into its individual words or tokens. It also includes identifying the base form of words (stemming or lemmatization).
Example: The sentence “The quick brown fox jumps over the lazy dog” would be tokenized into [“The”, “quick”, “brown”, “fox”, “jumps”, “over”, “the”, “lazy”, “dog”].

- Syntactic Analysis (Parsing): 
This stage checks the grammatical structure of a sentence. It involves parsing the sentence to ensure that it follows the rules of syntax.
Example: For the sentence “The quick brown fox jumps over the lazy dog,” syntactic analysis would confirm that “fox” is the subject, “jumps” is the verb, and “over the lazy dog” is the prepositional phrase.

- Semantic Analysis: 
This stage focuses on understanding the meaning of the text. It involves mapping syntactic structures to their corresponding meanings.
Example: In the sentence “The apple ate a banana,” semantic analysis would identify that the sentence is nonsensical because apples cannot eat.

- Discourse Integration: 
This stage considers the context of the sentence within a larger body of text. It ensures that the meaning of a sentence is consistent with the sentences before and after it.
Example: In the text “Jack is a bright student. He spends most of his time in the library,” discourse integration helps understand that “He” refers to “Jack.”

- Pragmatic Analysis: 
This final stage involves understanding the intended effect or purpose of the text. It considers the context and the speaker’s intent.
Example: The sentence “Can you pass the salt?” is understood as a request rather than a question about someone’s ability to pass the salt.

These stages work together to enable machines to process and understand human language effectively

# 2. Ambiguity in Natural Languages
Ambiguity in natural languages occurs when a word, phrase, or sentence has multiple meanings. This can make it challenging for both humans and machines to understand the intended meaning without additional context. Here are some common types of ambiguities with examples:

- Lexical Ambiguity:
Example: “The bank is closed.”
This sentence can mean that a financial institution is not open, or it could refer to the side of a river being inaccessible.

- Syntactic Ambiguity:
Example: “Visiting relatives can be boring.”
This can mean that the act of visiting relatives is boring, or it can mean that relatives who visit are boring.

- Semantic Ambiguity:
Example: “He saw the man with the telescope.”
This can mean that he used a telescope to see the man, or it can mean that he saw a man who had a telescope.

- Pragmatic Ambiguity:
Example: “Can you pass the salt?”
This can be interpreted as a literal question about one’s ability to pass the salt, or as a polite request for the salt to be passed.
Understanding and resolving these ambiguities is a significant challenge in NLP, requiring sophisticated algorithms and models to interpret the context correctly.

# 3. WordNet: 
is a large lexical database of English, developed at Princeton University. It groups English words into sets of synonyms called synsets, each representing a distinct concept. These synsets are interlinked by various semantic relations such as synonymy, antonymy, hyponymy (subordinate), and hypernymy (superordinate).

WordNet provides more than just lists of synonyms. It organizes words based on their meanings and the relationships between those meanings. For example, the word “car” is linked to “automobile” as a synonym, and both are connected to broader terms like "vehicle".

WordNet is widely used in natural language processing (NLP) and computational linguistics for tasks such as word sense disambiguation, information retrieval, and text analysis.

# 4. Affix: 
is a morpheme that is attached to a word stem to form a new word or word form. Affixes can modify the meaning or grammatical function of a word12.

> There are four main types of affixes:

- Prefixes: 
These are added to the beginning of a word. For example, “un-” in “unhappy” or “re-” in "redo".

- Suffixes: 
These are added to the end of a word. For example, “-ness” in “happiness” or “-ed” in "walked".

- Infixes: 
These are inserted into the middle of a word. Infixes are rare in English but can be found in some other languages.

- Circumfixes: 
These are added to both the beginning and the end of a word. Circumfixes are not common in English but are used in some other languages.

Affixes play a crucial role in the structure and meaning of words, helping to create new words and modify existing ones.

# 5. What is Lemmatization?
Lemmatization is the process of reducing words to their base or dictionary form, known as the lemma. This technique considers the context and the meaning of the words, ensuring that the base form belongs to the language’s dictionary. For example, the words “running,” “ran,” and “runs” are all lemmatized to the lemma “run.”

> How Lemmatization Works? Lemmatization involves several steps:

- Part-of-Speech (POS) Tagging: 
Identifying the grammatical category of each word (e.g., noun, verb, adjective).

- Morphological Analysis: 
Analyzing the structure of the word to understand its root form.

- Dictionary Lookup: 
Using a predefined vocabulary to find the lemma of the word. For example, the word “better” would be lemmatized to “good” if it is identified as an adjective, whereas “running” would be lemmatized to “run” if identified as a verb.

> Techniques in Lemmatization

- Rule-Based Lemmatization: 
Uses predefined grammatical rules to transform words. For instance, removing the “-ed” suffix from regular past tense verbs.

- Dictionary-Based Lemmatization: 
Looks up words in a dictionary to find their base forms.

- Machine Learning-Based Lemmatization: 
Employs machine learning models trained on annotated corpora to predict the lemma of a word.

> Benefits:
- Accuracy: 
Lemmatization provides more accurate results because it considers the context and meaning of words.
- Standardization: 
Ensures words are reduced to their dictionary form, aiding in tasks like text normalization and information retrieval.

> Limitations:

- Complexity: 
Requires more computational resources and a comprehensive dictionary.
- Dependency on POS Tagging: 
Requires accurate POS tagging, which adds to the processing overhead.

# 6. What is Stemming?

Stemming is a more straightforward process that cuts off prefixes and suffixes (i.e., affixes) to reduce a word to its root form. This root form, known as the stem, may not be a valid word in the language. For example, the words “running,” “runner,” and “runs” might all be stemmed to “run” or “runn,” depending on the stemming algorithm used.

> How Stemming Works?

Stemming algorithms apply a series of rules to strip affixes from words. The most common stemming algorithms include:

- Porter Stemmer: 
Uses a set of heuristic rules to iteratively remove suffixes.

- Snowball Stemmer: 
An extension of the Porter Stemmer with more robust rules.

- Lancaster Stemmer: 
A more aggressive stemmer that can sometimes over-stem words.
For example, the words “running”, “runner”, and “runs” might all be reduced to “run” by a stemming algorithm, but sometimes it might also reduce “arguing” to “argu”.

> Benefits:

- Simplicity: 
Stemming is straightforward and computationally inexpensive.
- Speed: 
Faster processing time due to simple rules and lack of context consideration.
- Useful in applications where speed is crucial, such as search engines

> Limitations:

- Accuracy: 
Can produce stems that are not actual words, leading to less accurate results.
- Over-Stemming: 
Can sometimes strip too much off a word (e.g., “running” to “runn”).
- Under-Stemming: 
Can sometimes strip too little off a word (e.g., “running” to “run”).

# 7. Training and Testing datasets
In machine learning, datasets are typically divided into two main subsets: training data and testing data. Here's a brief overview of each:

> Training Data

- Purpose:
The training data is used to train the machine learning model. It helps the model learn patterns, relationships, and features within the data.

- Process: 
During training, the model is exposed to the training data and adjusts its parameters to minimize errors in its predictions.

- Size: 
Typically, the training data constitutes a larger portion of the overall dataset, often around 60-80%.

> Testing Data

- Purpose: 
The testing data is used to evaluate the performance of the trained model. It helps determine how well the model generalizes to new, unseen data.

- Process: 
After training, the model is tested on the testing data to assess its accuracy, precision, recall, and other performance metrics.

- Size: 
The testing data usually makes up the remaining 20-40% of the dataset

> Importance of Separation

Avoiding Overfitting: By keeping the training and testing data separate, we ensure that the model does not simply memorize the training data but learns to generalize from it.

Performance Evaluation: Testing on unseen data provides a realistic measure of the model's performance in real-world scenarios

# 8. Part-of-Speech (POS) tagging: 
is a process in Natural Language Processing (NLP) where each word in a text is assigned a specific part of speech, such as noun, verb, adjective, etc., based on its context. This helps in understanding the grammatical structure and meaning of the sentence.

> Different Approaches to POS Tagging

- Rule-Based Tagging: 
Uses a set of hand-written linguistic rules.

- Statistical Tagging: 
Utilizes probabilistic models like Hidden Markov Models (HMM) and Conditional Random Fields (CRF).

- Transformation-Based Tagging: 
Also known as Brill Tagging, it applies transformation rules to improve initial tagging.

- Neural Network-Based Tagging: 
Employs deep learning models like Recurrent Neural Networks (RNN) and Long Short-Term Memory (LSTM) networks.

- Hybrid Tagging: 
Combines multiple approaches to leverage their strengths.

# 9. Rule-Based Tagging: 
is one of the earliest methods used for Part-of-Speech (POS) tagging. It relies on a set of hand-crafted linguistic rules to assign tags to words. Here’s a detailed explanation:

> How Rule-Based Tagging Works


- Lexical Lookup: 
Initially, each word in the text is assigned a list of possible tags based on a dictionary or lexicon. For example, the word “book” might be tagged as a noun (NN) or a verb (VB).

- Contextual Rules: 
The system then applies a series of contextual rules to narrow down the possible tags. These rules are based on the linguistic context of the words. For example:

    - If a word is preceded by a determiner (like “the” or “a”), it is likely to be a noun.

    - If a word ends in “ing” and is preceded by a verb, it is likely to be a present participle (VBG).

- Disambiguation: 
The rules are applied iteratively to resolve ambiguities and assign the most appropriate tag to each word. For instance:
“He can fish” vs. “He bought a can of fish.” In the first sentence, “can” is a verb, while in the second, it is a noun.

> Advantages:

- Simple and interpretable.
- Effective for languages with well-defined grammatical rules.

> Disadvantages:

- Requires extensive manual effort to create and maintain rules.
- May not handle ambiguous or unseen words well.
- Less effective for languages with complex or less rigid grammatical structures.

# 10. NLP Nymy'es
- Homonymy: 
Words that have the same spelling or pronunciation but different meanings.
    - Example: The word “bat” can mean a flying mammal or a piece of sports equipment used in cricket or baseball.

- Polysemy:
A single word with multiple related meanings.
    - Example: 
    The word “bank” can refer to the side of a river or a financial institution. Both meanings are related to the idea of a place where something is stored or accumulated.

- Synonymy:
Words that have the same or nearly the same meaning.
    - Example: 
    “Big” and “large” are synonyms because they both describe something of considerable size.

- Hyponymy:
A relationship where the meaning of one word is included within the meaning of another.
    - Example: 
    “Rose” is a hyponym of “flower” because a rose is a type of flower.

- Hypernymy:
The opposite of hyponymy; a word whose meaning includes the meanings of other words.
    - Example: 
    “Animal” is a hypernym of “dog,” “cat,” and “elephant” because all these are types of animals.

- Meronymy:
A relationship where one word denotes a part of something that is denoted by another word.
    - Example: 
    “Wheel” is a meronym of “car” because a wheel is a part of a car.

- Antonymy:
Words that have opposite meanings.
    - Example: 
    “Hot” and “cold” are antonyms because they describe opposite temperatures.