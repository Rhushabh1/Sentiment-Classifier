import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import WordNetLemmatizer
import pickle
import numpy as np 
import pandas as pd 

lemmatizer = WordNetLemmatizer()

'''
polarity 0 = negative, 2 = neutral, 4 = positive
id
date
query
user
tweet
'''

def init_process(fin, fout):
	outfile = open(fout, 'a')
	with open(fin, buffering = 200000, encoding = 'latin-1') as f:
		try:
			for line in f:
				line = line.replace('"', '')
				initial_polarity = line.split(',')[0]
				if initial_polarity =='0':
					initial_polarity = [1, 0]
				elif initial_polarity == '4':
					initial_polarity = [0, 1]

				tweet = line.split(',')[-1]
				outline = str(initial_polarity)+':::'+tweet
				outfile.write(outline)
		except Exception as e:
			print(str(e))
	outfile.close()

# if you need starting sets 
# RUN ONCE 
# init_process('training.1600000.processed.noemoticon.csv', 'train_set.csv')
# RUN ONCE
# init_process('testdata.manual.2009.06.14.csv', 'test_set.csv')

# ############################# #

def create_lexicon(fin):
	lexicon = []
	with open(fin, 'r', buffering = 100000, encoding = 'latin-1') as f:
		try:
			counter = 1
			content = ''
			for line in f:
				counter+=1 
				if (counter/2500.0).is_integer():
					tweet = line.split(':::')[1]
					content += ' '+tweet
					words = word_tokenize(content)
					words = [lemmatizer.lemmatize(i) for i in words]
					lexicon = list(set(lexicon + words))
					print(counter, len(lexicon))

		except Exception as e:
			print(str(e))

	with open('lexicon-2500-2638.pickle', 'wb') as f:
		pickle.dump(lexicon, f)


# ONLY IF YOU NEED LEXICON CREATED
# RUN ONCE
# create_lexicon('train_set.csv')

def convert_to_vec(fin, fout, lexicon_pickle):
	with open(lexicon_pickle, 'rb') as f:
		lexicon = pickle.load(f)
	outfile = open(fout, 'a')
	with open(fin, buffering=20000, encoding= 'latin-1') as f:
		counter = 0
		for line in f:
			counter+=1
			print(counter)

			label = line.split(':::')[0]
			tweet = line.split(':::')[1]

			current_words = word_tokenize(tweet.lower())
			current_words = [lemmatizer.lemmatize(i) for i in current_words]

			features = np.zeros(len(lexicon))

			for word in current_words:
				if word.lower() in lexicon:
					index_value = lexicon.index(word.lower())
					features[index_value] += 1

			features = list(features)
			outline = str(features)+'::'+str(label)+'\n'
			outfile.write(outline)

		print(counter)


# DON'T RUN THIS ONE, WAY TOO BIG, DO IN LINE
convert_to_vec('train_set.csv', 'processed-train-set-2500-2638.csv', 'lexicon-2500-2638.pickle')
# GO AHEAD AND RUN THIS ONE
# convert_to_vec('test_set.csv', 'processed-test-set-2500-2638.csv', 'lexicon-2500-2638.pickle')

def shuffle_data(fin):
	df = pd.read_csv(fin, error_bad_lines = False)
	df = df.iloc[np.random.permutation(len(df))]
	print(df.head())
	df.to_csv('train_set_shuffled.csv', index = False)

# RUN ONCE
# shuffle_data('train_set.csv')

def create_test_data_pickle(fin):

	featuresets = []
	labels = []
	counter = 0
	with open('processed-test-set-2500-2638.csv', buffering=20000) as f:
		for line in f:
			try:
				features = list(eval(line.split("::")[0]))
				label = list(eval(line.split('::')[1]))

				# print(features)
				# print(labels)
				
				featuresets.append(features)
				labels.append(label)
				counter+=1

			except:
				pass
	print(counter)
	featuresets = np.array(featuresets)
	labels =np.array(labels)

# RUN ONCE
# create_test_data_pickle('processed-test-set-2500-2638.csv')






