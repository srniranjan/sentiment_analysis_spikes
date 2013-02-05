from __future__ import division
from util.util import show_confusion_matrix
from util.util import write_probdist_for
import nltk
import json

stopwords = set(nltk.corpus.stopwords.words('english'))
nltk.config_megam('/Users/admin/Downloads/megam_0.92/megam')

def word_presence_features(document):
	features = {}
	for word in document:
		features['has(%s)' % word] = True
	return features

f = open('../../data_files/fs_tips.txt')
ip_json = json.load(f)
f.close()

feature_sets = []
for id, reviews in ip_json.iteritems():
	for review in reviews:
		review_words = [w for w in nltk.wordpunct_tokenize(review['text'].lower()) if w not in stopwords]
		feature_sets.append( (word_presence_features(review_words), review['sentiment']) )

separator = int(len(feature_sets) / 2.0)
train_set, test_set = feature_sets[:separator], feature_sets[separator:]
#classifier = nltk.NaiveBayesClassifier.train(train_set)
classifier = nltk.MaxentClassifier.train(train_set, algorithm='megam', trace=0)
print nltk.classify.accuracy(classifier, test_set)
classifier.show_most_informative_features(5)

show_confusion_matrix(classifier, test_set)
write_probdist_for(classifier, '../../results/prob_dist/word_presence.txt')
