# This script is based on  the sklearn_crfsuite tutorial by Mikhail Korobov

from itertools import chain

import nltk
import sklearn
import scipy.stats
from pprint import pprint

from sklearn.metrics import make_scorer
from sklearn.cross_validation import cross_val_score    # NB these two modules
                                                        # are deprecated
from sklearn.grid_search import RandomizedSearchCV

import sklearn_crfsuite
from sklearn_crfsuite import scorers
from sklearn_crfsuite import metrics



# Corpus: the CONLL 2002 NER corpus, Spanish data
train_sents = list(nltk.corpus.conll2002.iob_sents('esp.train'))
train_sents = train_sents[0:2]
test_sents = list(nltk.corpus.conll2002.iob_sents('esp.testb'))
test_sents = test_sents[0:2]

# define the features

def word2features(sent, i):
    word = sent[i][0]
    postag = sent[i][1]

    features = {
        'bias': 1.0,
        'word.lower()': word.lower(),
        'word[-3:]': word[-3:],
        'word[-2:]': word[-2:],
        'word.isupper()': word.isupper(),
        'word.istitle()': word.istitle(),
        'word.isdigit()': word.isdigit(),
        'postag': postag,
        'postag[:2]': postag[:2],
        }
    if i > 0:
        word1 = sent[i-1][0]
        postag1 = sent[i-1][1]
        features.update({
            '-1:word.lower()': word1.lower(),
            '-1:word.istitle()': word1.istitle(),
            '-1:word.isupper()': word1.isupper(),
            '-1:postag': postag1,
            '-1:postag[:2]': postag1[:2],
            })
    else:
        features['BOS'] = True

    if i < len(sent)-1:
        word1 = sent[i+1][0]
        postag1 = sent[i+1][1]
        features.update({
            '+1:word.lower()': word1.lower(),
            '+1:word.istitle()': word1.istitle(),
            '+1:word.isupper()': word1.isupper(),
            '+1:postag': postag1,
            '+1:postag[:2]': postag1[:2],
            })
    else:
        features['EOS'] = True

    return features

def sent2features(sent):
    return [word2features(sent, i) for i in range(len(sent))]

def sent2labels(sent):
    return [label for token, postag, label in sent]

def sent2tokens(sent):
    return [token for token, postag, label in sent]

# extract features from the data
X_train = [sent2features(s) for s in train_sents]
y_train = [sent2labels(s) for s in train_sents]
X_test = [sent2features(s) for s in test_sents]
y_test = [sent2labels(s) for s in test_sents]

# training a model
crf = sklearn_crfsuite.CRF(
     algorithm='lbfgs',
     c1=0.1,
     c2=0.1,
     max_iterations=100,
     all_possible_transitions=True
 )
crf.fit(X_train, y_train)

# Predict (after removing O)
labels = list(crf.classes_)
labels.remove('O')

pprint(X_test)
pprint("***")
y_pred = crf.predict(X_test)
pprint(y_pred)
metrics.flat_f1_score(y_test, y_pred,
                      average='weighted', labels=labels)