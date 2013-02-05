import nltk
import json

tuples = []
f = open('../../data_files/yelp_reviews.txt')
yelp_json = json.load(f)
f.close()
for id, tips in yelp_json.iteritems():
	for tip in tips:
		words = set(nltk.wordpunct_tokenize(tip['text'].lower()))
		for word in words:
			tuples.append( (word, tip['sentiment']) )
	
cfd = nltk.ConditionalFreqDist(tuples)
f = open('temp_pdndud252525.txt', 'w')

for word in cfd.keys():
	pc = cfd[word]['P']
	nc = cfd[word]['N']
	uc = cfd[word]['U']
	#	f.write( word.encode('utf-8') + "<<>>" + str(pc) + "<<>>" + str(nc) + "<<>>" + str(uc))
	if (pc + nc + uc) > 2:
		pd = ((pc - nc - uc) / (pc + nc + uc)) 
 		nd = ((nc - uc - pc) / (pc + nc + uc))
		ud = ((pc - uc - nc) / (pc + uc + nc))
		if pd > 0.25 or nd > 0.25 or ud > 0.25:
			f.write(word.encode('utf-8'))
			f.write('\n')

f.close()
	
