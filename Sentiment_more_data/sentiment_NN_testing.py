import tensorflow as tf 
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np 
import pickle
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()


n_nodes_hl1 = 500
n_nodes_hl2 = 500

############################
n_classes = 2
# 200 data? 52% (10 epochs)
# 2000 data? 62% (10 epochs)
# 2000 data? 63% (15 epochs)
# 200000 data? 74.3% (15 epochs)
hm_data = 2000000
###########################

batch_size = 32
hm_epochs = 10

x = tf.placeholder('float')
y = tf.placeholder('float')

current_epoch = tf.Variable(1)

hidden_1_layer = {'f_fum':n_nodes_hl1,
				  'weights':tf.Variable(tf.random_normal([2638, n_nodes_hl1])),
				  'biases':tf.Variable(tf.random_normal([n_nodes_hl1])) }
hidden_2_layer = {'f_fum':n_nodes_hl2,
                  'weights':tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),
				  'biases':tf.Variable(tf.random_normal([n_nodes_hl2])) }
output_layer = {'f_fum':None,
				'weights':tf.Variable(tf.random_normal([n_nodes_hl2, n_classes])),
				'biases':tf.Variable(tf.random_normal([n_classes])) }

# Nothing changes
def neural_network_model(data):

	l1 = tf.add(tf.matmul(data, hidden_1_layer['weights']) ,hidden_1_layer['biases'])
	l1 = tf.nn.relu(l1)

	l2 = tf.add(tf.matmul(l1, hidden_2_layer['weights']) ,hidden_2_layer['biases'])
	l2 = tf.nn.relu(l2)

	output = tf.matmul(l2, output_layer['weights']) + output_layer['biases']

	return output

# -------------IMPORTANT
# we initialize the saver object outside the session, but use it to restore the model inside the session
saver = tf.train.Saver()
# just to log the epochs
tf_log = 'tf.log'

def train_neural_network(x):
	prediction = neural_network_model(x)
	cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y) )
	optimizer = tf.train.AdamOptimizer(learning_rate = 0.001).minimize(cost)

	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())

		try:
			epoch = int(open(tf_log, 'r').read().split('\n')[-2])+1
			print('Starting: ', epoch)
		except:
			epoch = 1


		while epoch <= hm_epochs:
			if epoch != 1:
				saver.restore(sess, 'model.ckpt')
			epoch_loss = 1

			with open('lexicon-2500-2638.pickle', 'rb') as f:
				lexicon = pickle.load(f)

			with open('train_set_shuffled.csv', buffering = 20000, encoding = 'latin-1') as f:
				counter = 0
				for line in f:
					counter+=1
					# print(line)

					label = line.split(':::')[0]
					tweet = line.split(':::')[1]
					current_words = word_tokenize(tweet.lower())
					current_words = [lemmatizer.lemmatize(i) for i in current_words]

					features = np.zeros(len(lexicon))

					for word in current_words:
						if word.lower() in lexicon:
							index_value = lexicon.index(word.lower())
							features[index_value] += 1

					batch_x = np.array([list(features)])
					batch_y = np.array([eval(label)])

					# saver.save

					_, c = sess.run([optimizer, cost], feed_dict = {x:batch_x, y:batch_y})
					epoch_loss += c 
					# print(counter)

					if counter > hm_data:
						print('reached', hm_data, 'data, breaking')
						break 

					# i += batch_size

			# this part changes
			saver.save(sess, 'model.ckpt')
			print('Epoch', epoch, 'completed out of ', hm_epochs, 'loss: ', epoch_loss)

			# adding this
			# grab epoch as some sort of EXTERNAL value?
			# also log things like accuracy and loss!
			with open(tf_log, 'a') as f:
				f.write(str(epoch)+'\n')
			epoch += 1


		correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
		accuracy = tf.reduce_mean(tf.cast(correct, 'float'))

		featuresets = []
		labels = []
		counter =0
		with open('processed-test-set-2500-2638.csv', buffering = 20000) as f:
			for line in f:
				try:
					features = list(eval(line.split("::")[0]))
					label = list(eval(line.split('::')[1]))

					featuresets.append(features)
					labels.append(label)
					counter+=1

				except:
					pass

		print('Tested', counter, 'samples')

		test_x = np.array(featuresets)
		test_y = np.array(labels)


		print('Accuracy', accuracy.eval({x:test_x, y:test_y}))



# train_neural_network(x)

def test_neural_network():
	prediction = neural_network_model(x)

	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())

		for epoch in range(hm_epochs):
			try:
				saver.restore(sess, 'model.ckpt')
			except Exception as e:
				print(str(e))
			epoch_loss = 0

		correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
		accuracy = tf.reduce_mean(tf.cast(correct, 'float'))

		featuresets = []
		labels = []
		counter =0
		with open('processed-test-set-2500-2638.csv', buffering = 20000) as f:
			for line in f:
				try:
					features = list(eval(line.split("::")[0]))
					label = list(eval(line.split('::')[1]))

					featuresets.append(features)
					labels.append(label)
					counter+=1
				except:
					pass

		print('Tested', counter, 'samples')

		test_x = np.array(featuresets)
		test_y = np.array(labels)


		print('Accuracy', accuracy.eval({x:test_x, y:test_y}))


test_neural_network()