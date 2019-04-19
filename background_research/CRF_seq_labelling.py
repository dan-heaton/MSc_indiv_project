#Please note: the following is the implementation of building and using a CRF model for sequence labelling,
#as set out in 'Performing Sequence Labelling using CRF in Python'
#(http://www.albertauyeung.com/post/python-sequence-labelling-with-crf/) and is not my original work


from bs4 import BeautifulSoup as bs
from bs4.element import Tag
import codecs
import nltk
from sklearn.model_selection import train_test_split
import pycrfsuite as pcs
import numpy as np
from sklearn.metrics import classification_report


with codecs.open("datasets\\reuters.xml", "r", "utf-8") as infile:
    soup = bs(infile, "html5lib")

#Extract and create tuples for each token in every document and store in 'docs' list
docs = []
for elem in soup.find_all("document"):
    texts = []
    for c in elem.find("textwithnamedentities").children:
        if type(c) == Tag:
            if c.name == "namedentityintext":
                label = "N"
            else:
                label = "I"
            for w in c.text.split(" "):
                if len(w) > 0:
                    texts.append((w, label))
    docs.append(texts)

#Generate PoS tags to transform tuples from '(token, label)' to '(token, PoS tag, label)'
data = []
for i, doc in enumerate(docs):
    tokens = [t for t, label in doc]
    tagged = nltk.pos_tag(tokens)
    data.append([(w, pos, label) for (w, label), (word, pos) in zip(doc, tagged)])



def word2features(doc, i):
    """Takes a doc (in form of list of tuples above) and an index (ith document)
    and returns the document with features extracted"""
    word = doc[i][0]
    postag = doc[i][1]
    features = ['bias', 'word.lower='+word.lower(), 'word[-3:]='+word[-3:], 'word[-2:]='+word[-2:],
                'word.isupper=%s'%word.isupper(), 'word.istitle=%s'%word.istitle(),
                'word.isdigit=%s'%word.isdigit(), 'postag='+postag]

    if i > 0:
        word1 = doc[i-1][0]
        postag1 = doc[i-1][1]
        features.extend(['-1:word.lower='+word1.lower(), '-1:word.istitle=%s'%word1.istitle(),
                         '-1word.isupper=%s'%word1.isupper(), '-1:word.isdigit=%s'%word1.isdigit(), '-1:postag='+postag1])
    else:
        features.append('BOS')

    if i < len(doc)-1:
        word1 = doc[i+1][0]
        postag1 = doc[i+1][1]
        features.extend(['+1:word.lower='+word1.lower(), '+1:word.istitle=%s'%word1.istitle(),
                         '+1:word.isupper=%s'%word1.isupper(), '+1:word.isdigit=%s'%word1.isdigit(), '+1:postag='+postag1])
    else:
        features.append('EOS')

    return features



def extract_features(doc):
    return [word2features(doc, i) for i in range(len(doc))]


def get_labels(doc):
    return [label for (token, postag, label) in doc]


x = [extract_features(doc) for doc in data]
y = [get_labels(doc) for doc in data]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)


trainer = pcs.Trainer(verbose=True)

#Submits training data for one doc at a time (w/ all features extracted for each word)
for xseq, yseq in zip(x_train, y_train):
    trainer.append(xseq, yseq)

trainer.set_params({'c1': 0.1, 'c2': 0.01, 'max_iterations': 200, 'feature.possible_transitions': True})
trainer.train('crf.model')

tagger = pcs.Tagger()
tagger.open('crf.model')
y_pred = [tagger.tag(xseq) for xseq in x_test]

#Examines the predicted labels for the 13th document in the test set
i = 12
for x, y in zip(y_pred[i], [x[1].split("=")[1] for x in x_test[i]]):
    print("%s (%s)"%(y,x))


labels = {"N": 1, "I": 0}
#Converts each tag in each document to either a 1 or 0 and puts them all together in one array
predictions = np.array([labels[tag] for row in y_pred for tag in row])
truths = np.array([labels[tag] for row in y_test for tag in row])

print(classification_report(truths, predictions, target_names=["I", "N"]))