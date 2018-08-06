
import pickle 

# list of unique hindi words from fb Embeddings 
with open("wiki.hi.vec") as file :
	line = file.readlines()
vectors = [x.strip()for x in line]

hindi_words = []
for sentence in line :
	words = sentence.split()
	hindi_words.append(words[0])

file = open("hindi_words.txt", "w")
for word in hindi_words:
	file.write("%s\n" %word)


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
		print(split_vector[0], len(split_vector))

pickle_out = open("hindi_embeddings.pickle", "wb")
pickle.dump(word_embeddings, pickle_out)
pickle_out.close()







