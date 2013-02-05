import nltk
import codecs

def show_confusion_matrix(classifier, test_set):
	gold_tags = [tag for (sent, tag) in test_set]
	test_tags = [classifier.classify(sent) for (sent, tag) in test_set]
	print nltk.ConfusionMatrix(gold_tags, test_tags)

def write_probdist_for(classifier, file_name):
	f = codecs.open(file_name, 'w', 'utf-8')
	for (tag, word), pd in classifier._feature_probdist.iteritems():
		f.write(tag + ":" + word + ":" + str(pd.prob(True)) + ":" + str(pd.prob(False)))
		f.write('\n')
	f.close()
