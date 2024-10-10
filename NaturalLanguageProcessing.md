// TO BE DONE


11. Explain edit distance algorithm with an example. Show working of the minimum no of operations required to transform “kitten” into  “sitting”.

// DONE

2. Discuss various challenges in processing natural languages.

3. Explain Porter’s Stemming algorithm with  example.

4. Explain with suitable example following relationships between word meanings: Homonymy, Polysemy, Synonymy,Hyponomy,Hypernomy,Meronomy,Antonomy.

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

# 1.1 Challenges in NLP: 

due to the complexity and diversity of human language. Here are some of the key challenges:

- Ambiguity: 
Words and sentences can have multiple meanings depending on the context. For example, the word “bank” can refer to a financial institution or the side of a river. Resolving such ambiguities is a significant challenge.
    - Homonyms: 

- Contextual Understanding: 
Understanding the context in which words are used is crucial for accurate interpretation. This includes recognizing idiomatic expressions, sarcasm, and cultural references.

- Data Quality and Quantity: 
High-quality, annotated data is essential for training NLP models. However, obtaining such data can be expensive and time-consuming. Additionally, some languages and dialects have limited available data.

- Multilingualism and Language Variation: 
There are thousands of languages and dialects, each with its own grammar, vocabulary, and cultural nuances. Developing NLP systems that can handle multiple languages and variations is a complex task.

- Syntax and Grammar Complexity: 
Human languages have intricate syntactic structures and grammatical rules. Parsing these structures accurately is challenging, especially for languages with free word order or complex morphology.

- Semantic Knowledge Representation: 
Capturing the meaning of words and sentences in a way that machines can understand is difficult. This involves understanding relationships between words, such as synonyms, antonyms, and hierarchical relationships.

- Real-Time Processing and Efficiency: 
Many NLP applications, such as chatbots and voice assistants, require real-time processing. Ensuring that NLP models are efficient and can handle large volumes of data quickly is a significant challenge.

- Errors in Text & Speech
Text Errors: Misspelled words or missing words can hinder text analysis.
Mispronunciations and different accents can pose challenges for understanding spoken language.


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
is a morpheme that is attached to a word stem to form a new word or word form. Affixes can modify the meaning or grammatical function of a word.

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
Uses a set of heuristic rules to iteratively remove suffixes. Porter’s stemming algorithm is a widely used method in natural language processing (NLP) for reducing words to their root form, known as the “stem.” This process helps in normalizing text data, making it easier to analyze and process.

    > How Porter’s Stemming Algorithm Works
    The algorithm works by applying a series of rules to remove common suffixes from words. These rules are applied in a specific order to ensure that the stemming process is consistent and accurate. Here are the main steps:

        - Step 1a: Remove common suffixes like “sses” -> “ss”, “ies” -> “i”, “ss” -> “ss”, “s” -> “”.
        - Step 1b: Remove “eed” -> “ee” if the word has a vowel before “eed”; otherwise, remove “ed” or “ing” if the word has a vowel     before them.
        - Step 1c: If the word ends with “y” and has a vowel before it, replace “y” with “i”.
        - Step 2: Replace suffixes like “ational” -> “ate”, “tional” -> “tion”, etc.
        - Step 3: Replace suffixes like “icate” -> “ic”, “ative” -> “”, “alize” -> “al”, etc.
        - Step 4: Remove suffixes like “al”, “ance”, “ence”, “er”, “ic”, etc., if the word has a vowel before them.
        - Step 5: Remove “e” if the word has a vowel before it, and remove “l” if the word ends with “ll”.

    > Example: Let’s take the word “running” as an example:

        - Step 1a: No change, as “running” does not end with “sses”, “ies”, “ss”, or “s”.
        - Step 1b: “running” ends with “ing” and has a vowel before it, so “ing” is removed, leaving “runn”.
        - Step 1c: No change, as “runn” does not end with “y”.
        - Step 2: No applicable suffixes in “runn”.
        - Step 3: No applicable suffixes in “runn”.
        - Step 4: No applicable suffixes in “runn”.
        - Step 5: No change, as “runn” does not end with “e” or “ll”.
    > So, the stemmed form of “running” is “run”.

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


> # UNIT - 2

# 1. Word Sense Disambiguation (WSD): 
is a process in natural language processing (NLP) that determines which meaning of a word is being used in a given context. Many words have multiple meanings, and WSD helps to identify the correct one based on the surrounding text. For example, the word “bank” can refer to a financial institution or the side of a river. WSD uses contextual clues to disambiguate such words. This technique is crucial for improving the accuracy of various NLP applications like machine translation, sentiment analysis, and information retrieval


# 2. ML approch of WSD:

The Naive Bayes approach for Word Sense Disambiguation (WSD) is a supervised machine learning method that uses probabilistic models to determine the most likely sense of a word based on its context. Here’s a step-by-step explanation of how it works:

- Training Data: 
The model is trained on a corpus of text where words are annotated with their correct senses. This annotated data is crucial for learning the relationships between words and their contexts.

- Feature Extraction: 
For each instance of an ambiguous word, features are extracted from the surrounding context. Common features include neighboring words, part-of-speech tags, and syntactic dependencies.

- Probability Calculation: 
The Naive Bayes classifier calculates the probability of each possible sense of the word given the extracted features. It assumes that the features are conditionally independent, which simplifies the computation.

- Sense Assignment: 
The sense with the highest probability is assigned to the ambiguous word. This is done using Bayes’ theorem:<br>
`P(sense ∣ context) = P(context ∣ sense) * P(sense)​ / P(context)` <br>
Here, `P(sense|context)` is the probability of a sense given the context, `P(context|sense)` is the probability of the context given the sense, `P(sense)` is the prior probability of the sense, and `P(context)` is the probability of the context.

- Evaluation: 
The performance of the Naive Bayes classifier is evaluated using metrics like accuracy, precision, recall, and F1-score on a separate test set.

The Naive Bayes approach is popular due to its simplicity and effectiveness, especially when there is a large amount of annotated training data available

# 3. Named Entity Recognition (NER): 
is a crucial task in Natural Language Processing (NLP) that involves identifying and classifying key elements in text into predefined categories. These categories typically include names of people, organizations, locations, dates, quantities, monetary values, and more. NER helps in extracting meaningful information from unstructured text, making it easier to analyze and understand large volumes of data.

## Applications: 
It is widely used in various fields such as information retrieval, question answering, machine translation, and text summarization.

## Techniques: 
Modern NER systems often use machine learning models, including deep learning techniques like transformers, to achieve high accuracy.

## Challenges: 
NER must handle ambiguities and variations in language, such as different spellings, abbreviations, and context-dependent meanings.

## Example:
In the sentence `“Apple Inc. is looking at buying U.K. startup for $1 billion,”` NER would identify:
Apple Inc. as an Organization
U.K. as a Location
$1 billion as a Monetary Value
NER is a foundational tool in NLP that enhances the ability to process and understand human language by focusing on the most relevant parts of the text.

# 4. WordNet: 
is a large lexical database of English, developed at Princeton University. It groups English words into sets of synonyms called synsets, provides short definitions and usage examples, and records various semantic relations between these synonym sets.
## Key Features:
- Synsets: 
Collections of synonymous words that express the same concept.
- Semantic Relations: 
Includes hypernyms (general terms), hyponyms (specific terms), meronyms (part-whole relationships), and antonyms (opposites).
- Applications: 
Widely used in NLP tasks such as word sense disambiguation, information retrieval, and machine translation.
- Structure: 
Organized into nouns, verbs, adjectives, and adverbs, each forming a separate network of meaningfully related words.

# 5. Reference Resolution: 
is a fundamental task in Natural Language Processing (NLP) that involves determining what entities pronouns and other referring expressions in a text refer to. This task is crucial for understanding and generating coherent text.
`Types of References:` Includes pronouns (he, she, it), definite descriptions (the car), and demonstratives (this, that).
`Challenges:` Ambiguity in language, varying contexts, and the need for world knowledge to resolve references accurately.
## Techniques:
- Rule-based Methods:
 Use linguistic rules to resolve references.
- Machine Learning Approaches:
 Employ algorithms trained on annotated corpora to predict references.
- Deep Learning Models:
 Utilize neural networks, especially transformers, to improve accuracy by capturing complex patterns in data.

`Example:`
In the sentence “John took his dog to the park because it was a sunny day,” reference resolution identifies “his” as referring to “John” and “it” as referring to “the park.”

# 6. Machine Translation (MT): 
is the process of using artificial intelligence to automatically translate text or speech from one language to another. It leverages natural language processing (NLP) and deep learning techniques to understand and generate translations.

## Key Approaches:
- Rule-Based Machine Translation (RBMT): 
Uses predefined linguistic rules and dictionaries to translate text. This method is less common today due to its limitations in handling complex language structures.
- Statistical Machine Translation (SMT): 
Relies on statistical models derived from large bilingual text corpora. It predicts translations based on the probability of word sequences.
- Neural Machine Translation (NMT): 
Utilizes neural networks, particularly deep learning models, to provide more accurate and fluent translations. NMT models, like those used by Google Translate, can capture context better and handle nuances in language.

## Applications:
Real-Time Translation: Tools like Google Translate and Microsoft Translator enable real-time translation of text and speech, facilitating communication across languages.
Content Localization: Helps businesses translate websites, documents, and software to cater to global audiences.
Multilingual Customer Support: Enhances customer service by providing support in multiple languages without the need for human translators.
## Challenges:
Context and Ambiguity: Accurately translating idiomatic expressions, cultural references, and context-dependent meanings remains challenging.
Quality and Fluency: Ensuring translations are not only accurate but also natural and fluent.



