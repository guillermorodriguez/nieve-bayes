"""
    @ Author:       Guillermo Rodriguez
    @ Date:         09/18/2018
    @ Purpose:      Apply tokenization, lemmatization, and stemming to a data set.
                    Common point for the calculation of TF / IDF statistics.
    @ Dependency:   NLTK
                        python
                        >>> import nltk
                        >>> nltk.download()
"""

# python
# import nltk
# nltk.download()
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
import math

class statistics:

    """
        Constructor
    """
    def __init__(self):
        print('Statistics Object Initialized')
        self.vocabulary = []

    """
        @ Author:       Guillermo Rodriguez
        @ Date:         09/18/2018
        @ Purpose:      Tokenize a data set
    """
    def tokenization(self, data):
        result = {}

        for _word in data.split(' '):
            _word = _word.strip()

            if _word.encode('utf-8') in result:
                result[_word.replace('\n', '').encode('utf-8')] += 1
            else:
                result[_word.replace('\n', '').encode('utf-8')] = 1

            if _word.encode('utf-8') not in self.vocabulary:
                self.vocabulary.append(_word.replace('\n', '').encode('utf-8'))

        return result

    def probabilityOfValueInClass(self, data):
        result = {}
        class_count_total = 0

        for value in data.values():
            class_count_total += value

        for key, value in data.items():
            result[key] = ( value + 1 ) / ( class_count_total + len(self.vocabulary))

        return result

    def posteriorProbability(self, data, class_probability, spam, not_spam, other):
        result = {}

        for key, value in data.items():
            denominator = 0
            if key in spam:
                denominator += spam[key]
            if key in not_spam:
                denominator += not_spam[key]
            if key in other:
                denominator += other[key]

            denominator /= ( sum(spam.values()) + sum(not_spam.values()) + sum(other.values()) )

            if denominator == 0:
                denominator = 1

            result[key] = ( value * class_probability ) / denominator

        return result
