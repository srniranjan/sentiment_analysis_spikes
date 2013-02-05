from __future__ import division
import nltk
import json
from synset_parser import imp_words
from nltk.corpus import wordnet as wn
import codecs

f = open('../../data_files/fs_tips_sents.txt')
ip_json = json.load(f)
f.close()

def get_sentiment_for(sent):
	words = set(nltk.wordpunct_tokenize(sent))
	pc, nc = 0, 0
	ps, ns = 0, 0
	for word in words:
		if word in imp_words:
			if imp_words[word][0] > imp_words[word][1]:
				ps += imp_words[word][0]
				pc += 1
				f.write(word + " has positive sentiment of " + str(ps) + '\n')
			else:
				ns += imp_words[word][1]
				nc += 1
				f.write(word + " has negative sentiment of " + str(ns) + '\n')
	pd  = (ps / pc) if pc > 0 else 0
	nd =  (ns / nc) if nc > 0 else 0
	senti_dist = (abs(pd - nd) / (pd + nd)) if (pd + nd) > 0 else 0
	if senti_dist > 0.3 and pd > nd:
		return 'P'
	elif senti_dist > 0.3 and nd > pd:
		return 'N'
	return 'U'

f = codecs.open('../../results/simple_sents_parser_output.txt', 'w', 'utf-8')
gold_tags = []
test_tags = []
for id, reviews in ip_json.iteritems():
	for review in reviews:
		test_tag = None
		num_pos = 0
		num_neg = 0
		for sent in review['text']:
			f.write(sent + '\n')
			sentiment = get_sentiment_for(sent)
			if sentiment == 'P':
				num_pos += 1
			elif sentiment == 'N':
				num_neg += 1
			f.write('+++++++++\n')
		if num_pos > num_neg:
			test_tag = 'P'
		elif num_neg > num_pos:
			test_tag = 'N'
		else:
			test_tag = 'U'
		gold_tags.append(review['sentiment'])
		test_tags.append(test_tag)
		f.write(review['sentiment'] + " ---> " + test_tag + "\n")
		f.write('~~~~~~~~~~~~\n')
f.close()

print nltk.ConfusionMatrix(gold_tags, test_tags)
