# learn how to preprocess data that will fit in out neural_network model

import nltk
from nltk.tokenize import word_tokenize

'''-------------------TOKENIZE
i pulled the chair up to the table

[i, pulled, the, chair, up, to , the, table]
'''

from nltk.stem import WordNetLemmatizer

'''------------------LEMMATIZE
grouping different words with same meaning into a single word

[running, ran, run, runny, runs] = run 
'''

import numpy as np 
import pickle
import random
from collections import Counter

lemmatizer = WordNetLemmatizer()
hm_lines = 10000000


'''---------------LEXICON
[chair, table, spoon, television] = lexicon

i pulled a chair up to the table

[1, 1, 0, 0]
'''

def create_lexicon(pos, neg):
	lexicon = []
	for file in [pos, neg]:
		with open(file, 'r') as f:
			contents = f.readlines()
			for l in contents[:hm_lines]:
				all_words = word_tokenize(l.lower())
				lexicon += list(all_words)

	# lemmatize all of these words
	lexicon = [lemmatizer.lemmatize(i) for i in lexicon]
	word_counts = Counter(lexicon)
	# word_counts = {'the': 433, 'and':434}

	l2 = []
	for w in word_counts:
		# we don't want supper_common_words and also not so rare_words
		if 1000>word_counts[w] > 50:
			l2.append(w)

	print(len(l2))
	return l2


def sample_handling(sample, lexicon, classification):
	featureset = []
	'''
	[
	[[0, 1, 0, 1, 0], [1, 0]] # positive data 
	[[1 0, 1, , 1], [0, 1]] # negative data
	]
	'''

	with open(sample, 'r') as f:
		contents = f.readlines()
		for l in contents[:hm_lines]:
			current_words = word_tokenize(l.lower())
			current_words = [lemmatizer.lemmatize(i) for i in current_words]
			features = np.zeros(len(lexicon))
			for word in current_words:
				if word.lower() in lexicon:
					index_value = lexicon.index(word.lower())
					features[index_value] += 1
			features = list(features)
			featureset.append([features, classification])

	return featureset


def create_featureset_and_labels(pos, neg, test_size = 0.1):
	lexicon = create_lexicon(pos, neg)
	features = []
	# features will have all pos and neg samples
	features += sample_handling(pos, lexicon, [1, 0])
	features += sample_handling(neg, lexicon, [0, 1])
	random.shuffle(features)
	# shuffling is important because thats how neural_network works

	# basically the model asks a single question
	# tf.argmax([99999999999, -999999999999]) == tf.argmax([1, 0]) ??
	# so if we shuffle nicely then the question will probably be 
	# tf.argmax([5753, 3423]) == tf.argmax([1, 0]) ??

	features = np.array(features)

	testing_size = int(test_size*len(features))

	train_x = list(features[:,0][:-testing_size])
	train_y = list(features[:,1][:-testing_size])

	test_x = list(features[:,0][-testing_size:])
	test_y = list(features[:,1][-testing_size:])

	return train_x, train_y, test_x, test_y


if __name__ == '__main__':
	train_x, train_y, test_x, test_y = create_featureset_and_labels('positive.txt', 'negative.txt')

	with open('sentiment_set.pickle', 'wb') as f:
		pickle.dump([train_x, train_y, test_x, test_y], f)













