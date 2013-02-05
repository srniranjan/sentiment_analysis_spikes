from __future__ import division
import re

imp_words = {}
f = open('../../data_files/imp_words_uniq.txt')
lines = f.readlines()
f.close()
for line in lines:
	if line.startswith('#'):
		pass
	else:
		cols = line.rstrip().split('\t')
		pos_score = float(cols[0])
		neg_score = float(cols[1])
		imp_words[cols[2]] = (pos_score, neg_score)
