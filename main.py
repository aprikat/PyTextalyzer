from flask import Flask
from flask import render_template
from datetime import datetime

import nltk.classify.util
from nltk.classify import NaiveBayesClassifier
from nltk.corpus import movie_reviews
from sklearn.feature_extraction.text import TfidfVectorizer

app = Flask(__name__)

class Conversation:
    def __init__(self):
        self.inbound = []
        self.outbound = []

# class TextMessage:
#     def __init__(self):
#         self.sender = ''
#         self.timestamp = datetime.now()     # ?
#         self.body = ''


@app.route('/')
def index():

    with open('data/ankita.txt') as f:
        ankita_conv = Conversation()
        for line in f:
            line = line.strip()
            items = line.split('|')
            if items[0] == '+14088215768':
                tm = {}
                tm["sender"] = items[0]
                tm["timestamp"] = datetime.strptime(items[1], '%b %d, %Y, %I:%M %p')
                tm["body"] = items[2].strip()
                ankita_conv.outbound.append(tm)
            elif items[0] == '+15103649591':
                tm = {}
                tm["sender"] = items[0]
                tm["timestamp"] = datetime.strptime(items[1], '%b %d, %Y, %I:%M %p')
                tm["body"] = items[2].strip()
                ankita_conv.inbound.append(tm)
            else:
                pass

    # sentiment_classifier = sentiment_train()
    # print sentiment_classifier.classify(word_feats("Happy excellent magnificent outstanding"))

    print get_cosine_similarity(ankita_conv)


    return render_template('index.html')

def word_feats(words):
    return dict([(word, True) for word in words])

def sentiment_train():
    negids = movie_reviews.fileids('neg')
    posids = movie_reviews.fileids('pos')

    negfeats = [(word_feats(movie_reviews.words(fileids=[f])), 'neg') for f in negids]
    posfeats = [(word_feats(movie_reviews.words(fileids=[f])), 'pos') for f in posids]

    negcutoff = len(negfeats)*3/4
    poscutoff = len(posfeats)*3/4

    trainfeats = negfeats[:negcutoff] + posfeats[:poscutoff]
    testfeats = negfeats[negcutoff:] + posfeats[poscutoff:]
    # print 'train on %d instances, test on %d instances' % (len(trainfeats), len(testfeats))

    classifier = NaiveBayesClassifier.train(trainfeats)
    # print 'accuracy:', nltk.classify.util.accuracy(classifier, testfeats)
    # classifier.show_most_informative_features()
    return classifier

def get_cosine_similarity(conv):
    inbound = ""
    outbound = ""
    for tm in conv.inbound:
        # inbound.append(tm["body"])
        inbound += tm["body"] + " "
    for tm in conv.outbound:
        # outbound.append(tm["body"])
        outbound += tm["body"] + " "

    # print inbound
    # print outbound

    vect = TfidfVectorizer(min_df=1, decode_error='ignore')
    tfidf = vect.fit_transform([inbound, outbound])
    return (tfidf * tfidf.T).A

if __name__ == '__main__':
    app.debug = True
    app.run()