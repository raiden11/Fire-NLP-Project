import pickle 

# list of unique malyalam words from fb Embeddings 
with open("wiki.ml.vec") as file :
	line = file.readlines()
vectors = [x.strip()for x in line]

# malyalam_words = []
# for sentence in line :
# 	words = sentence.split()
# 	malyalam_words.append(words[0])

# file = open("malyalam_words.txt", "w")
# for word in malyalam_words:
# 	file.write("%s\n" %word)


# generate binary of word embeddings
word_embeddings = {}
del(vectors[0])

for vector in vectors:
	feature = []
	split_vector = vector.split()

	# remove this anomaly
	if len(split_vector) == 301:
		for i in range(1, 301):
			feature.append(float(split_vector[i]))
		word_embeddings[split_vector[0]] = feature
	else:
		print(len(split_vector))
		
pickle_out = open("malyalam_embeddings.pickle", "wb")
pickle.dump(word_embeddings, pickle_out)
pickle_out.close()

