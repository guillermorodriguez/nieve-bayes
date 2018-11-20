import os
import argparse
from graph import *
from statistics import *
import math

print('Started ....')

parser = argparse.ArgumentParser(prog='start.py')
parser.add_argument('-graph', help='Graphical Data Output [NO | YES]')
parser.add_argument('-statistics', help='Statistical Data Output [NO | YES]')
parse = parser.parse_args()


if parse.graph and parse.statistics:
    _path = os.getcwd() + '\\data\\'
    spam_token = {}                     # Tokenization of Data
    spam_token_test = {}
    spam_class = 0                      # Total spam documents

    not_spam_token = {}                 # Tokenization of Not spam
    not_spam_token_test = {}
    not_spam_class = 0                  # Total not spam documents

    other_token = {}                    # Tokenization of Other Emails
    other_token_test = {}
    other_class = 0                     # Other documents

    _statistics = statistics()

    # Spam - Read Data Set
    spam = []
    for _file in os.listdir(_path + "spam"):
        with open(_path + "spam/" + _file, encoding="latin-1") as spam_file:
            spam.append(spam_file.read())

    spam_class = math.floor(len(spam) * 0.1)
    spam_token_test = _statistics.tokenization(' '.join(spam[:spam_class]))
    spam_token = _statistics.tokenization(' '.join(spam[spam_class:]))

    not_spam = []
    for _file in os.listdir(_path + "not_spam"):
        with open(_path + "not_spam/" + _file, encoding="latin-1") as not_spam_file:
            not_spam.append(not_spam_file.read())

    not_spam_class = math.floor(len(not_spam)*0.1)
    not_spam_token_test = _statistics.tokenization(' '.join(not_spam[:not_spam_class]))
    not_spam_token = _statistics.tokenization(' '.join(not_spam[not_spam_class:]))


    other = []
    for _file in os.listdir(_path + "other"):
        with open(_path + "other/" + _file, encoding="latin-1") as other_file:
            other.append(other_file.read())

    other_class = math.floor(len(other)*0.1)
    other_token_test = _statistics.tokenization(' '.join(other[:other_class]))
    other_token = _statistics.tokenization(' '.join(other[other_class:]))

    print("Tokens:")
    print(spam_token_test)
    print(not_spam_token_test)
    print(other_token_test)

    print("Class Spam: %f" % ( spam_class / (spam_class + not_spam_class + other_class) ) )
    print("Class Not Spam: %f" % ( not_spam_class / (spam_class + not_spam_class + other_class) ) )
    print("Class Other: %f" % ( other_class / (spam_class + not_spam_class + other_class) ) )

    print("Vocabulary: %i" % len(_statistics.vocabulary))

    print("Conditional Probabilities")
    _out = _statistics.probabilityOfValueInClass(spam_token_test)
    posterior_spam = _statistics.posteriorProbability(_out, spam_class / (spam_class + not_spam_class + other_class), spam_token_test, not_spam_token_test, other_token_test)
    with open("spam.txt", 'w') as file:
        for key, value in posterior_spam.items():
            file.write("%s,%f\n" % (key, value))
    _graph = graph()
    _graph.create_plot(posterior_spam)

    _out = _statistics.probabilityOfValueInClass(not_spam_token_test)
    with open("not_spam.txt", 'w') as file:
        for key, value in _out.items():
            file.write("%s,%f\n" % (key, value))

    _out = _statistics.probabilityOfValueInClass(other_token_test)
    with open("other.txt", 'w') as file:
        for key, value in _out.items():
            file.write("%s,%f\n" % (key, value))



else:
    parser.print_help()

print('Completed ....')
