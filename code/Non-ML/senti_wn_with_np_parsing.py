import nltk
import json
from nltk.tag.simplify import simplify_wsj_tag
from synset_parser import imp_words
from nltk.corpus import wordnet as wn
import codecs
import re

grammar = r"""
	NP: {<DET|ADJ|ADV|PRO><U|DET|WH|ADJ|ADV|V|:|VN|TO|P|,|VG|VD>*<N|NP|:|ADJ|PRO>+}
	N: {<N|NP|ADJ|PRO>+}
"""
cp = nltk.RegexpParser(grammar)

f = open('../../data_files/fs_tips_sents.txt')
ip_json = json.load(f)
f.close()

f = open('../../data_files/amplifier_words.txt')
amplifier_words = {line.split('\t')[1].rstrip() : line.split('\t')[0] for line in f.readlines()}
f.close()

f = open('../../data_files/negation_words.txt')
negation_words = [w.rstrip() for w in f.readlines()]
f.close()

def pos_tag(sent):
	sent_pos = nltk.pos_tag(nltk.wordpunct_tokenize(sent.lower()))
	simplified = []
	for w, pos in sent_pos:
		if simplify_wsj_tag(pos):
			simplified.append( (w, simplify_wsj_tag(pos)) )
		else:
			simplified.append( (w, 'U') )
	return simplified

def get_sentiment_score_for(t):
	ps, pc = 0, 0
	ns, nc = 0, 0
	amp_fact, amp_count = 0, 0
	negation_words_present = False
	probable_subjects = []
	for (word, pos) in t.leaves():
		if pos == 'P' and word != 'love':
			continue
		if word in amplifier_words:
			amp_count = 1
			amp_val = float(amplifier_words[word])
			amp_fact = (amp_fact + amp_val) / amp_count
			continue
		elif word in negation_words:
			negation_words_present = True
			continue
		elif word in imp_words:
			if imp_words[word][0] > imp_words[word][1]:
				ps += imp_words[word][0]
				pc += 1
			else:
				ns += imp_words[word][1]
				nc += 1
			continue
		if pos == 'N' or pos == 'NP':
			probable_subjects.append(word)
	if pc > 0:
		ps = ps / pc
		ps = ps + (ps * amp_fact)
	if nc > 0:
		ns = ns / nc
		ns = ns + (ns * amp_fact)
	if negation_words_present:
		temp = ps
		ps = ns
		ns = temp
	return ps, ns, probable_subjects

def is_food_concept(subjects):
	for sub in subjects:
		for ss in wn.synsets(sub):
			paths = ss.hypernym_paths()
			for path in paths:
				for hyp_ss in path:
					if hyp_ss.name.startswith('food'):
						return True
	return False

def get_sentiment_for(sent):
	left_over_words = []
	simplified = pos_tag(sent.lower().encode('utf-8'))
	f.write( str(simplified) + '\n' )
	tree = cp.parse(simplified)
	max_ps, max_ns = 0, 0
	for t in tree:
		if type(t) == nltk.tree.Tree:
			curr_ps, curr_ns, probable_subjects = get_sentiment_score_for(t)
			f.write(str(t) + ":\n")
			f.write("	has positive sentiment of " + str(curr_ps) + '\n')
			f.write("	has negative sentiment of " + str(curr_ns) + '\n')
			f.write(" 	about: " + str(probable_subjects) + '\n')
			if is_food_concept(probable_subjects):
				f.write("	(A food concept!)\n")
			if curr_ps > max_ps:
				max_ps = curr_ps
			if curr_ns > max_ns:
				max_ns = curr_ns
		elif type(t) == tuple:
			word = t[0]
			if word in imp_words and (t[1] != 'P' or word == 'love'):
				if imp_words[word][0] > imp_words[word][1]:
					if imp_words[word][0] > max_ps:
						max_ps = imp_words[word][0]
					f.write(word + ' --- has positive sentiment of ' + str(imp_words[word][0]) + '\n')
				elif imp_words[word][1] > imp_words[word][0]:
					if imp_words[word][1] > max_ns:
						max_ns = imp_words[word][1]
					f.write(word + ' --- has negative sentiment of ' + str(imp_words[word][1]) + '\n')
	senti_dist = abs(max_ps - max_ns) / (max_ps + max_ns) if (max_ps + max_ns) > 0 else 0
	if senti_dist > 0.3 and max_ps > max_ns:
		return 'P'
	elif senti_dist > 0.3 and max_ns > max_ps:
		return 'N'
	return 'U'

def sanitize_words_in(sent):
	sent = sent.lower().replace("don't", "dont").replace("can't", "cant").replace("won't", "wont").replace("isn't", "isnt").replace("wouldn't", "wouldnt").replace("doesn't", "doesnt").replace("didn't", "didnt").replace("couldn't", "couldnt")
	sent = re.sub("y+u+m+", "yum", sent)
	sent = re.sub("y+u+m+y+", "yum", sent)
	sent = re.sub("w+a+y+", "way", sent)
	sent = re.sub("s+o+", "so", sent)
	sent = re.sub("s+w+e+t+", "sweet", sent)
	sent = re.sub("mm+", "mmm", sent)
	return sent 

f = open('../../results/sents_parser_output.txt', 'w')
gold_tags = []
test_tags = []
cm = {}
for id, reviews in ip_json.iteritems():
	for review in reviews:
		test_tag = None
		sentiments = []
		for sent in review['text']:
			sent = sanitize_words_in(sent)
			sentiment = get_sentiment_for(sent)
			f.write(sentiment + '\n')
			sentiments.append(sentiment)
			f.write('+++++++++\n')
		fd = nltk.FreqDist(sentiments)
		if fd['P'] > 0 and fd['N'] == 0:
			test_tag = 'P'
		elif fd['N'] > 0 and fd['P'] == 0:
			test_tag = 'N'
		elif fd['P'] > fd['N'] and fd['P'] > fd['U']:
			test_tag = 'P'
		elif fd ['N'] > fd['P'] and fd['N'] > fd['U']:
			test_tag = 'N'
		else:
			test_tag = 'U'
		gold_tags.append(review['sentiment'])
		test_tags.append(test_tag)
		cm_key = review['sentiment'] + " ---> " + test_tag 
		if cm_key not in cm:
			cm[cm_key] = []
		cm[cm_key].append(review['text'])
		f.write(review['sentiment'] + " ---> " + test_tag + "\n")
		f.write('~~~~~~~~~~~~\n')
f.close()

print nltk.ConfusionMatrix(gold_tags, test_tags)
f = codecs.open('../../results/sents_parser_confusion_matrix.txt', 'w')
for key, reviews in cm.iteritems():
	f.write('-----------\n')
	f.write(key + '\n')
	f.write('-----------\n')
	for review in reviews:
		for sent in review:
			f.write(str(sent.encode('utf-8')) + '\n')
		f.write('+++++++\n')
f.close()
