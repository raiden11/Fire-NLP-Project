

# import nltk
import tensorflow as tf
import tflearn
import random
import numpy as np
import random 
import operator 
import pickle 

language = "malyalam"
acronym = "ml"

# get_data
with open("v1_train." + acronym) as file :
	line = file.readlines()

words_data = [x.strip() for x in line]


with open("v1_test2." + acronym) as file :
	line = file.readlines()
test_words = [x.strip() for x in line]

with open("v1_test1." + acronym) as file :
	line = file.readlines()
test_words1 = [x.strip() for x in line]

# hindi_word_embeddings has word vectors for all availaible hindi words
pickle_in = open(language + "_embeddings.pickle", "rb")
word_embeddings = pickle.load(pickle_in)
dataset = []

classes = ["other", "number", "event", "datenum", "things", "organization", "occupation", "name", "location"]

# make dataset
def prepare_dataset(rows):

	i = 0
	for word in words_data:

		words = word.split()
		if words[0] != "newline":
			if words[0] in word_embeddings:
				vector = word_embeddings[words[0]]
			else :
				vector = np.random.uniform(low=-1.0, high=1.0, size=(300,))

			output = [0, 0, 0, 0, 0, 0, 0, 0, 0]
			if words[1] == "other":
				output[0] = 1
			elif words[1] == "number":
				output[1] = 1
			elif words[1] == "event":
				output[2] = 1
			elif words[1] == "datenum":
				output[3] = 1
			elif words[1] == "things":
				output[4] = 1
			elif words[1] == "organization":
				output[5] = 1
			elif words[1] == "occupation":
				output[6] = 1
			elif words[1] == "name":
				output[7] = 1
			elif words[1] == "location":
				output[8] = 1  
			else:
				print("hello")
				exit(0)
			dataset.append([vector, output, words[0].isdigit()])

		i += 1
		if i >= rows:
			break

	random.shuffle(dataset)


# divide dataset
def divide_dataset(ratio):

	train_length = (len(dataset)*ratio)//(1+ratio)
	train_set = dataset[ : train_length]
	test_set = dataset[train_length + 1 : ]

	train_x = list((np.array(train_set))[:, 0])
	train_y = list((np.array(train_set))[:, 1])

	return train_set, test_set, train_x, train_y


# Train model
def train_model(epochs, batchsize, train_x, train_y, rows):

	tf.reset_default_graph()

	net = tflearn.input_data(shape = [None, len(train_x[0])])
	net = tflearn.fully_connected(net, 8)
	net = tflearn.fully_connected(net, len(train_y[0]), activation = 'softmax')
	net = tflearn.regression(net)

	model = tflearn.DNN(net, tensorboard_dir = 'tflearn_logs')
	model.fit(train_x, train_y, n_epoch=epochs, batch_size=batchsize, show_metric=True)


	# testing on test file
	with open("out.txt", "w") as f:

		cnt = 0 
		for word in test_words:

			if word != "newline":

				if word.isdigit():
					if len(word) == 4:
						print(classes[3], file = f)
					else:
						print(classes[1], file = f)
					continue

				elif word in word_embeddings:
					features = word_embeddings[word]
				else:
					features = np.random.uniform(low=-1.0, high=1.0, size=(300,))

				prediction = model.predict([np.array(features)]).tolist()
				predicted_class, value = max(enumerate(prediction[0]), key=operator.itemgetter(1))

				print(classes[predicted_class], file=f)
			
			else:
				print("newline", file=f)

			cnt += 1	
			if cnt > rows : 
				break

	with open("out1.txt", "w") as f:

		cnt = 0 
		for word in test_words1:

			if word != "newline":

				if word.isdigit():
					if len(word) == 4:
						print(classes[3], file = f)
					else:
						print(classes[1], file = f)
					continue

				elif word in word_embeddings:
					features = word_embeddings[word]
				else:
					features = np.random.uniform(low=-1.0, high=1.0, size=(300,))

				prediction = model.predict([np.array(features)]).tolist()
				predicted_class, value = max(enumerate(prediction[0]), key=operator.itemgetter(1))

				print(classes[predicted_class], file=f)
			
			else:
				print("newline", file=f)

			cnt += 1	
			if cnt > rows : 
				break


	return model


# Test accuracy 
def calculate_test_accuracy(test_set, model):

	test_correct_prediction_count = [0]*9
	test_original_class_count = [0]*9
	test_prediction_count = [0]*9
	test_f1_score = [0]*9
	f1_score_sum = 0
	valid_classes = 0
	average_f1 = 0
	correct = 0
	total = 0

	for row in test_set :
		
		if row[2] == True:
			test_prediction_count[1] += 1
			test_correct_prediction_count[1] += 1
			test_original_class_count[1] += 1
			total += 1
			correct += 1
			continue

		prediction = model.predict([np.array(row[0])]).tolist()
		predicted_class, value = max(enumerate(prediction[0]), key=operator.itemgetter(1))
		test_prediction_count[predicted_class] += 1
		
		if row[1][predicted_class] == 1:
			correct += 1
			test_correct_prediction_count[predicted_class] += 1
		total += 1
		original_class, value = max(enumerate(row[1]), key=operator.itemgetter(1))
		test_original_class_count[original_class] += 1

	test_recall = [0]*9
	test_precision = [0]*9
	for i in range(0, 9):
		if test_original_class_count[i] == 0:
			test_recall[i] = -1
		else:
			test_recall[i] = (float(test_correct_prediction_count[i])/float(test_original_class_count[i]))*100
		
		if test_prediction_count[i] == 0:
			test_precision[i] = -1
		else:	
			test_precision[i] = (float(test_correct_prediction_count[i])/float(test_prediction_count[i]))*100

		if test_precision[i] == -1 or test_recall[i] == -1 :
			test_f1_score[i] = -1
		elif test_precision[i] == 0 and test_recall[i] == 0:
			test_f1_score[i] = -1
		else: 
			test_f1_score[i] = (2*test_recall[i]*test_precision[i])/(test_precision[i] + test_recall[i])
			f1_score_sum += test_f1_score[i] 
			valid_classes += 1

	print("\nTEST SET : ")
	print("correct : ", correct, ", total : ", total)
	print("accuracy : ", (correct/total)*100)
	print("recall : ", test_recall)
	print("Precision :", test_precision)
	print("count in class : ", test_original_class_count)
	print("Class F1 Score : ", test_f1_score)
	print("Average F1 Score : ", f1_score_sum/valid_classes)

# Train Accuracy
def calculate_train_accuracy(train_set, model):
	
	train_correct_prediction_count = [0]*9
	train_original_class_count = [0]*9
	train_prediction_count = [0]*9
	train_f1_score = [0]*9
	f1_score_sum = 0
	valid_classes = 0
	average_f1 = 0
	correct = 0
	total = 0

	for row in train_set :
		
		if row[2] == True:
			train_prediction_count[1] += 1
			train_correct_prediction_count[1] += 1
			train_original_class_count[1] += 1
			total += 1
			correct += 1
			continue	 
		
		prediction = model.predict([np.array(row[0])]).tolist()
		predicted_class, value = max(enumerate(prediction[0]), key=operator.itemgetter(1))
		train_prediction_count[predicted_class] += 1
		
		if row[1][predicted_class] == 1:
			correct += 1
			train_correct_prediction_count[predicted_class] += 1
		total += 1
		original_class, value = max(enumerate(row[1]), key=operator.itemgetter(1))
		train_original_class_count[original_class] += 1

	train_recall = [0]*9
	train_precision = [0]*9
	for i in range(0, 9):
		if train_original_class_count[i] == 0:
			train_recall[i] = -1
		else:
			train_recall[i] = (float(train_correct_prediction_count[i])/float(train_original_class_count[i]))*100
		
		if train_prediction_count[i] == 0:
			train_precision[i] = -1
		else:
			train_precision[i] = (float(train_correct_prediction_count[i])/float(train_prediction_count[i]))*100

		if train_precision[i] == -1 or train_recall[i] == -1 :
			train_f1_score[i] = -1
		elif train_precision[i] == 0 and train_recall[i] == 0:
			train_f1_score[i] = -1
		else:
			train_f1_score[i] = (2*train_recall[i]*train_precision[i])/(train_precision[i] + train_recall[i])
			f1_score_sum += train_f1_score[i]
			valid_classes += 1

	print("\nTRAIN SET : ")
	print("correct : ", correct, "total : ", total)
	print("accuracy : ", (correct/total)*100)
	print("recall : ", train_recall)
	print("Precision : ", train_precision)
	print("count in class : ", train_original_class_count)
	print("Class F1 Score : ", train_f1_score)
	print("Average F1 Score : ", f1_score_sum/valid_classes)


if __name__ == "__main__":

	parameters = {}
	parameters["epoch"] = 10
	parameters["rows"] =  10000000
	parameters["batchsize"] = 8 
	parameters["ratio"] = 20000000

	prepare_dataset(parameters["rows"])

	train_set, test_set, train_x, train_y = divide_dataset(parameters["ratio"])
	model = train_model(parameters["epoch"], parameters["batchsize"], train_x, train_y, 100000000)

	print("Epochs : ", parameters["epoch"], " Words : ", parameters["rows"] )
	print("Size of Training Set : ", len(train_set))
	print("Size of Test Set : ", len(test_set))

	calculate_test_accuracy(test_set, model)
	calculate_train_accuracy(train_set, model)

	# test_parameters = {}
	# test_parameters["rows"] = 1000
	# test_parameters["model_name"] = "site-test-1.tflearn"
	# test_model(test_parameters["model_name"], test_parameters["rows"], train_x, train_y)





































