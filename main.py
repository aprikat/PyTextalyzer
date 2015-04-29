from flask import Flask
from flask import render_template
from flask import jsonify
from datetime import datetime

import nltk.classify.util
from nltk.classify import NaiveBayesClassifier
from nltk.corpus import movie_reviews
from nltk.model.ngram import NgramModel
from nltk.probability import LidstoneProbDist
from sklearn.feature_extraction.text import TfidfVectorizer

from textblob import TextBlob

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

def read_texts(textfile, sender):
    with open(textfile) as f:
        this_conv = Conversation()
        for line in f:
            line = line.strip()
            items = line.split('|')
            if items[0] == '+14088215768':
                tm = {}
                tm["sender"] = items[0]
                # tm["timestamp"] = datetime.strptime(items[1], '%b %d, %Y, %I:%M %p')
                tm["timestamp"] = items[1].decode('utf-8', errors='ignore')
                tm["body"] = items[2].strip()
                this_conv.outbound.append(tm)
            elif items[0] == sender:
                tm = {}
                tm["sender"] = items[0]
                # tm["timestamp"] = datetime.strptime(items[1], '%b %d, %Y, %I:%M %p')
                tm["timestamp"] = items[1].decode('utf-8', errors='ignore')
                tm["body"] = items[2].strip()
                this_conv.inbound.append(tm)
            else:
                pass
    f.close()
    return this_conv


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/ankita')
def ankita():
    ankita_conv = read_texts('data/ankita.txt', '+15103649591')
    inbound_sentiments = []
    outbound_sentiments = []

    # sentiment analysis
    inbound_sentiments, outbound_sentiments = get_sentiments(ankita_conv)

    # trigram modeling
    # lm = generate_language_model(ankita_conv.inbound)
    # print lm

    # cosine similarity
    matrix = get_cosine_similarity(ankita_conv)[0]
    sim = str(matrix).split(' ')[-1][:-1]

    return render_template('friend.html',
        name='Ankita Agharkar',
        your_sentiments=outbound_sentiments,
        friend_sentiments=inbound_sentiments,
        cosine_sim=sim)


@app.route('/riley')
def riley():
    riley_conv = read_texts('data/riley.txt', '+18186323954')
    inbound_sentiments = []
    outbound_sentiments = []

    # sentiment analysis
    inbound_sentiments, outbound_sentiments = get_sentiments(riley_conv)

    # cosine similarity
    matrix = get_cosine_similarity(riley_conv)[0]
    sim = str(matrix).split(' ')[-1][:-1]

    return render_template('friend.html',
        name='Riley Pietsch',
        your_sentiments=outbound_sentiments,
        friend_sentiments=inbound_sentiments,
        cosine_sim=sim)


@app.route('/christina')
def christina():
    christina_conv = read_texts('data/christina.txt', '+19165217921')
    inbound_sentiments = []
    outbound_sentiments = []

    # sentiment analysis
    inbound_sentiments, outbound_sentiments = get_sentiments(christina_conv)

    # cosine similarity
    matrix = get_cosine_similarity(christina_conv)[0]
    sim = str(matrix).split(' ')[-1][:-1]

    return render_template('friend.html',
        name='Christina Milanes',
        your_sentiments=outbound_sentiments,
        friend_sentiments=inbound_sentiments,
        cosine_sim=sim)

@app.route('/mom')
def mom():
    mom_conv = read_texts('data/mom.txt', '+14088211126')
    inbound_sentiments = []
    outbound_sentiments = []

    # sentiment analysis
    inbound_sentiments, outbound_sentiments = get_sentiments(mom_conv)

    # cosine similarity
    matrix = get_cosine_similarity(mom_conv)[0]
    sim = str(matrix).split(' ')[-1][:-1]

    return render_template('friend.html',
        name='Helene Deng',
        your_sentiments=outbound_sentiments,
        friend_sentiments=inbound_sentiments,
        cosine_sim=sim)


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


def get_sentiments(conv):
    inbound = []
    outbound = []

    for tm in conv.inbound:
        timestamp = tm["timestamp"] # datetime.datetime
        body = tm["body"].decode('ascii', errors='ignore')
        score = TextBlob(body).sentiment[0]
        inbound.append([timestamp, score])

    for tm in conv.outbound:
        timestamp = tm["timestamp"] # datetime.datetime
        body = tm["body"].decode('ascii', errors='ignore')
        score = TextBlob(body).sentiment[0]
        outbound.append([timestamp, score])

    inbound = list(inbound)
    outbound = list(outbound)
    return (inbound, outbound)


# Get the cosine similarity between two participants in a text conversation
def get_cosine_similarity(conv):
    inbound = ""
    outbound = ""
    for tm in conv.inbound:
        inbound += tm["body"] + " "
    for tm in conv.outbound:
        outbound += tm["body"] + " "

    vect = TfidfVectorizer(min_df=1, decode_error='ignore')
    tfidf = vect.fit_transform([inbound, outbound])
    return (tfidf * tfidf.T).A


# Given a list of a person's tokenized texts, generate a language model
def generate_language_model(texts):
    tokenized = []
    for t in texts:
        body = t["body"].split(" ")
        for b in body:
            tokenized.append(b)

    est = lambda fdist : LidstoneProbDist(fdist, 0.2)
    lm = NgramModel(3, tokenized, estimator=est)
    return lm

if __name__ == '__main__':
    app.debug = True
    app.run()