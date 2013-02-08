import json
import nltk

f = open('../../data_files/fs_tips.txt')
yelp_json = json.load(f)
f.close()

op_json = {}
for id, reviews in yelp_json.iteritems():
	op_json[id] = []
	for review in reviews:
		rev_json = {}
		rev_json['text'] = []
		rev_json['sentiment'] = review['sentiment']
		for sent in nltk.sent_tokenize(review['text']):
			rev_json['text'].append(sent)
		op_json[id].append(rev_json)

f = open('../../data_files/fs_tips_sents.txt', 'w')
json.dump(op_json, f)
f.close()
