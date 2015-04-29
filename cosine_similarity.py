from sklearn.feature_extraction.text import TfidfVectorizer

import nltk.corpus


class CosineSimilarity:

	def __init__(self):
		self.vect = TfidfVectorizer(min_df=1, decode_error='ignore')
		self.corpora = { nltk.corpus.brown : "brown", 
			nltk.corpus.gutenberg : "gutenberg", 
			nltk.corpus.webtext : "webtext", 
			nltk.corpus.reuters : "reuters",
			nltk.corpus.inaugural : "inaugural" }

	def _calculate_cosine_similarity(self, corpus1, corpus2):
		tfidf = self.vect.fit_transform([corpus1, corpus2])
		A = (tfidf * tfidf.T).A
		matrix = A[0]
		sim = str(matrix).split(' ')[-1][:-1]
		return sim

	# Get the cosine similarity between two participants in a text conversation
	def get_conv_cosine_similarity(self, conv):
	    inbound = ""
	    outbound = ""
	    for tm in conv.inbound:
	        inbound += tm["body"] + " "
	    for tm in conv.outbound:
	        outbound += tm["body"] + " "

	    return self._calculate_cosine_similarity(inbound, outbound)

	def get_cosine_similarity_to_corpora(self, outbound_conv):
		outbound = ""
		for tm in outbound_conv:
			# outbound.append(tm["body"])
			outbound += tm["body"] + " "

		similarities = {}

		for corpus, name in self.corpora.iteritems():
			corp_concat = " ".join(corpus.words())
			corp_sim = self._calculate_cosine_similarity(corp_concat, outbound)
			similarities[name] = corp_sim

		return similarities










