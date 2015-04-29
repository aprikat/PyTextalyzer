from flask import Flask
from flask import render_template
from flask import jsonify
from datetime import datetime

import nltk.classify.util
from nltk.classify import NaiveBayesClassifier
from nltk.corpus import movie_reviews
from nltk.probability import LidstoneProbDist

from textblob import TextBlob

from markov import Markov
from cosine_similarity import CosineSimilarity

app = Flask(__name__)

class Conversation:
    def __init__(self):
        self.inbound = []
        self.outbound = []

cs = CosineSimilarity()

# class TextMessage:
#     def __init__(self):
#         self.sender = ''
#         self.timestamp = datetime.now()     # ?
#         self.body = ''

def read_texts(textfile, sender, plain_text):
    # ptxt_out = open(plain_text_out, 'w+')
    # ptxt_in = open(plain_text_in, 'w+')
    ptxt = open(plain_text, 'w+')
    with open(textfile) as f:
        this_conv = Conversation()
        for line in f:
            line = line.strip()
            items = line.split('|')
            # outbound
            if items[0] == '+14088215768':
                tm = {}
                tm["sender"] = items[0]
                # tm["timestamp"] = datetime.strptime(items[1], '%b %d, %Y, %I:%M %p')
                tm["timestamp"] = items[1].decode('utf-8', errors='ignore')
                tm["body"] = items[2].strip()
                # ptxt_out.write(tm["body"] + " ")
                ptxt.write(tm["body"] + " ")
                this_conv.outbound.append(tm)
            # inbound
            elif items[0] == sender:
                tm = {}
                tm["sender"] = items[0]
                # tm["timestamp"] = datetime.strptime(items[1], '%b %d, %Y, %I:%M %p')
                tm["timestamp"] = items[1].decode('utf-8', errors='ignore')
                tm["body"] = items[2].strip()
                # ptxt_in.write(tm["body"] + " ")
                ptxt.write(tm["body"] + " ")
                this_conv.inbound.append(tm)
            else:
                pass
    f.close()
    # ptxt_out.close()
    # ptxt_in.close()
    ptxt.close()
    return this_conv


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/ankita')
def ankita():
    ankita_conv = read_texts('data/ankita.txt', '+15103649591', 'data/ankita_plain.txt')
    inbound_sentiments = []
    outbound_sentiments = []

    # language modeling
    sample_sentence = generate_markov('data/ankita_plain.txt')

    # sentiment analysis
    inbound_sentiments, outbound_sentiments = get_sentiments(ankita_conv)

    # trigram modeling
    # lm = generate_language_model(ankita_conv.inbound)
    # print lm

    # cosine similarity
    sim = cs.get_conv_cosine_similarity(ankita_conv)
    # sim = str(matrix).split(' ')[-1][:-1]

    # map of corpora to their cosine similarities
    corpora_sims = cs.get_cosine_similarity_to_corpora(ankita_conv.outbound)

    return render_template('friend.html',
        name='Ankita Agharkar',
        sentence=sample_sentence,
        your_sentiments=outbound_sentiments,
        friend_sentiments=inbound_sentiments,
        brown_sim=corpora_sims["brown"],
        cosine_sim=sim)


@app.route('/riley')
def riley():
    riley_conv = read_texts('data/riley.txt', '+18186323954', 'data/riley_plain.txt')
    inbound_sentiments = []
    outbound_sentiments = []

    # language modeling
    sample_sentence = generate_markov('data/riley_plain.txt')

    # sentiment analysis
    inbound_sentiments, outbound_sentiments = get_sentiments(riley_conv)

    # cosine similarity
    matrix = get_conv_cosine_similarity(riley_conv)[0]
    sim = str(matrix).split(' ')[-1][:-1]

    return render_template('friend.html',
        name='Riley Pietsch',
        sentence=sample_sentence,
        your_sentiments=outbound_sentiments,
        friend_sentiments=inbound_sentiments,
        cosine_sim=sim)


@app.route('/christina')
def christina():
    christina_conv = read_texts('data/christina.txt', '+19165217921', 'data/christina_plain.txt')
    inbound_sentiments = []
    outbound_sentiments = []

    # language modeling
    sample_sentence = generate_markov('data/christina_plain.txt')

    # sentiment analysis
    inbound_sentiments, outbound_sentiments = get_sentiments(christina_conv)

    # cosine similarity
    matrix = get_conv_cosine_similarity(christina_conv)[0]
    sim = str(matrix).split(' ')[-1][:-1]

    return render_template('friend.html',
        name='Christina Milanes',
        sentence=sample_sentence,
        your_sentiments=outbound_sentiments,
        friend_sentiments=inbound_sentiments,
        cosine_sim=sim)

@app.route('/mom')
def mom():
    mom_conv = read_texts('data/mom.txt', '+14088211126', 'data/mom_plain.txt')
    inbound_sentiments = []
    outbound_sentiments = []

    # language modeling
    sample_sentence = generate_markov('data/mom_plain.txt')

    # sentiment analysis
    inbound_sentiments, outbound_sentiments = get_sentiments(mom_conv)

    # cosine similarity
    matrix = get_conv_cosine_similarity(mom_conv)[0]
    sim = str(matrix).split(' ')[-1][:-1]

    return render_template('friend.html',
        name='Helene Deng',
        sentence=sample_sentence,
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

# Custom markov stuff!
def generate_markov(textfile):
    f = open(textfile)
    markov_model = Markov(f)
    f.close()
    return markov_model.generate_markov_text(size=16)

if __name__ == '__main__':
    app.debug = True
    app.run()