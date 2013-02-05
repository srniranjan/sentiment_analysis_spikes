from __future__ import division
import datetime
import sys
import nltk
import pickle
import json
from util.util import show_confusion_matrix

def read_file(name):
	ret_val = None
	with open(name, 'rb') as input:
		ret_val = pickle.load(input)
	return ret_val

def doc_word_presence(document):
	doc_words = set([(w.lower(), pos) for w, pos in document])
	features = {}
	for word, pos in most_suggestive_words:
		features[(word, pos)] = ((word, pos) in doc_words)
	return features

print 'Reading most frequent words...', datetime.datetime.now()
most_freq_pos_words = read_file('../../data_files/yelp_pos_words')
most_freq_neg_words = read_file('../../data_files/yelp_neg_words')
most_freq_unk_words = read_file('../../data_files/yelp_unk_words')
most_suggestive_words = most_freq_pos_words.union(most_freq_neg_words).union(most_freq_unk_words)

f = open('../../data_files/yelp_sent_pos_text.txt')
lines = f.readlines()
f.close()
tips_pos = []
for line in lines:
	tips_pos.append( eval(line) )

print 'Building feature set...', datetime.datetime.now()
count = 0
featuresets = []
for tip, tag in tips_pos:
	features = doc_word_presence(tip)
	featuresets.append( (features, tag) )
	count += 1
	if count == int(len(tips_pos)):
		break

size = int(len(featuresets)/2)
train_set, test_set = featuresets[:size], featuresets[size:]
print 'Training classifier...', datetime.datetime.now()
#classifier = nltk.NaiveBayesClassifier.train(train_set)
nltk.config_megam('/Users/admin/Downloads/megam_0.92/megam')
classifier = nltk.MaxentClassifier.train(train_set, algorithm='megam')
print 'Finished training classifier', datetime.datetime.now()
print nltk.classify.accuracy(classifier, test_set)

show_confusion_matrix(classifier, test_set)
write_probdist_for(classifier, '../../results/prob_dist/top_word_presence.txt')
