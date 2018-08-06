import tensorflow as tf
import pickle 
import random
import numpy as np
from tensorflow import keras


with open("v1_train.hi") as file :
	line = file.readlines()

hindi_words = [x.strip() for x in line]

pickle_in = open("hindi_embeddings.pickle", "rb")
word_embeddings = pickle.load(pickle_in)
dataset = []
classes = ["other", "number", "event", "datenum", "things", "organization", "occupation", "name", "location", "zero"]

# dataset contains a number of sentences and each sentence has 100 words each
# For a sentence, we have 3 items in list : [[list of 100 vectors(of length 300)], [list of output vectors(of length 10)], [is digit or not(Boolean)] ]
def prepare_dataset(rows):

	count = 0
	sentence = []
	digit_or_not = []
	sentence_output = []
	word_count = 0

	for word in hindi_words:

		output = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
		words = word.split()
		if words[0] != "newline":
			
			if words[0] in word_embeddings:
				vector = word_embeddings[words[0]]
			else :
				vector = np.random.uniform(low=-1.0, high=1.0, size=(300,))

			for i in range(0, 9):
				if words[1] == classes[i]:
					output[i] = 1
			word_count += 1

			sentence.append(vector)
			sentence_output.append(output)
			digit_or_not.append(words[0].isdigit())
			
		else :

			print(word_count)
			while word_count < 100:	

				output[9] = 1
				word_count += 1
				vector = [0]*300
				sentence.append(vector)
				sentence_output.append(output)
				digit_or_not.append(False)

			dataset.append([sentence, sentence_output, digit_or_not])
			
			word_count = 0
			sentence = []
			digit_or_not = []
			sentence_output = []

		count += 1
		if count >= rows:
			break

	# random.shuffle(dataset)


def prepare_dataset_1(rows):

	word_count = 0
	sentence = []
	count = 0

	for word in hindi_words:

		output = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
		words = word.split()
		if words[0] != "newline":
			
			if words[0] in word_embeddings:
				vector = word_embeddings[words[0]]
			else :
				vector = np.random.uniform(low=-1.0, high=1.0, size=(300,))

			for i in range(0, 9):
				if words[1] == classes[i]:
					output[i] = 1
			word_count += 1

			# sentence.append([vector, output, words[0].isdigit()])
			sentence.append([vector, output, words[0]])
			
		else :

			while word_count < 100:	

				output[9] = 1
				word_count += 1
				vector = [0]*300
				sentence.append([vector, output, False])

			for row in sentence:
				dataset.append(row)
			# print(len(dataset))
			# dataset.append(sentence)
			
			word_count = 0
			sentence = []

		count += 1
		if count >= rows:
			break

test_words = []
train_words = []

def divide_dataset(ratio) :

	train_length = int((((len(dataset)*ratio)/100)//(1+ratio))*100)

	train_set = dataset[:train_length]
	test_set = dataset[train_length : ]
	
	# print(train_set[0])

	train_x = []
	test_x = []
	train_y = []
	test_y = []

	for row in train_set:
		train_words.append(row[2])
		if type(row[0]) is not list:
			train_x.append(row[0].tolist())
		else:
			train_x.append(row[0])
		if type(row[1]) is not list:
			train_y.append(row[1].tolist())
		else:
			train_y.append(row[1])

	for row in test_set:
		test_words.append(row[2])
		if type(row[0]) is not list:
			test_x.append(row[0].tolist())
		else:
			test_x.append(row[0])
		if type(row[1]) is not list:
			test_y.append(row[1].tolist())
		else:
			test_y.append(row[1])


	return train_set, test_set, train_x, train_y, test_x, test_y


def train_model(epoch, batchsize, train_x, train_y, test_x, test_y):

	data = np.random.random((100, 10))
	# labels = np.random.random((100, 2))

	model = keras.Sequential()
	model.add(keras.layers.Dense(64, activation='relu'))
	model.add(keras.layers.Dense(64, activation='relu'))
	model.add(keras.layers.Dense(10, activation='softmax'))

	model.compile(optimizer=tf.train.AdamOptimizer(0.001),
              loss='categorical_crossentropy',
              metrics = ['accuracy'])

	# print(data)
	# print(train_y)
	# print("hello")

	# print(data)
	model.fit(np.array(train_x), np.array(train_y), epochs=50, batch_size=100)
	# model.evaluate(np.array(test_x), np.array(test_y), batch_size=100, verbose = 2)
	vals = model.predict(np.array(test_x), batch_size=100, verbose=1)
	
	i = 0
	for val in vals :
		index = np.argmax(val)
		if index != 9:
			print(test_words[i], classes[index])
		i += 1

	# return model

	# return model

	# data = tf.placeholder(tf.float32, [None, 100, 300]) # No. of sentences, words in sentence, dimension of word
	# target = tf.placeholder(tf.float32, [None, 10])
	# num_hidden = 24
	# cell = tf.nn.rnn_cell.LSTMCell(num_hidden, state_is_tuple = True)
	# val, _ = tf.nn.dynamic_rnn(cell, data, dtype=tf.float32)
	# val = tf.transpose(val, [1, 0, 2])
	# last = tf.gather(val, int(val.get_shape()[0]) - 1)
	# weight = tf.Variable(tf.truncated_normal([num_hidden, int(target.get_shape()[1])]))
	# bias = tf.Variable(tf.constant(0.1, shape=[target.get_shape()[1]]))
	
	# prediction = tf.nn.softmax(tf.matmul(last, weight) + bias)
	# cross_entropy = -tf.reduce_sum(target * tf.log(tf.clip_by_value(prediction,1e-10,1.0)))

	# optimizer = tf.train.AdamOptimizer()
	# minimize = optimizer.minimize(cross_entropy)

	# mistakes = tf.not_equal(tf.argmax(target, 1), tf.argmax(prediction, 1))
	# error = tf.reduce_mean(tf.cast(mistakes, tf.float32))
	


if __name__ == "__main__":

	parameters = {}
	parameters["epoch"] = 2
	parameters["rows"] = 20000
	parameters["batchsize"] = 8 
	parameters["ratio"] = 9
	prepare_dataset_1(parameters["rows"])
	train_set, test_set, train_x, train_y, test_x, test_y = divide_dataset(parameters["ratio"])
	
	# print(train_x[0])
	# print(train_y[0])

	train_model(parameters["epoch"], parameters["batchsize"], train_x, train_y, test_x, test_y)


	# print(len(dataset))
	# print(len(dataset[0]))
	# print(len(dataset[1]))
	# print(len(dataset[2]))
	# print(len(dataset[3]))
	# print(len(dataset[4]))
	# print(dataset[0][90])
	# print(dataset[2][90])
	# print(dataset[3][80])
	# print(dataset[4][70])
	# print(dataset[2][50])

	# train_set, test_set, train_x, train_y = divide_dataset(parameters["ratio"])



