1. Compare derivational and inflectional morphology with suitable example.
2. Discuss various challenges in processing natural languages.
3. Explain Porter’s Stemming algorithm with  example.
4. Explain with suitable example following relationships between word meanings: Homonymy, Polysemy, Synonymy,Hyponomy,Hypernomy,Meronomy,Antonomy.
5. What is Natural language processing (NLP)? Discuss various stages involved in NLP process with suitable example.
6. Explain N-gram model with example.


8. What is Natural language processing? Explain ambiguity in Natural languages with suitable examples.
10. What is POS tagging? List different approaches of POS tagging. Explain anyone in brief.
11. Explain edit distance algorithm with an example. Show working of the minimum no of operations required to transform “kitten” into  “sitting”.


13. Write a short note on Wordnet.
14. What is the difference between stemming and lemmitization?
15. Write a short note on Training and Testing of dataset.
16. What is POS tagging? What are the challenges faced by POS tagging?
17. Describe open and closed class words in english with examples.
18. Define affixes . Explain the types of affixes.
20. Differentiate between syntactic ambiguity and lexical ambiguity.
21. Explain the role of FSA in morphological analysis.



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

# 1. Natural Language Processing (NLP): 
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