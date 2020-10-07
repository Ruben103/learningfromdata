import json
from numpy import array
from sklearn.preprocessing import label_binarize

class Data

    def get_embedding(word, embeddings):
	try:
		# GloVe embeddings only have lower case words
		return embeddings[word.lower()]
	except KeyError:
		return embeddings['UNK']

# Load noun-noun compound data
def load_data():
	print("Loading data...")
	# Embeddings
	embeddings = json.load(open('embeddings.json', 'r'))
	# Training and development data
	X_train = []
	Y_train = []
	with open('training_data.tsv', 'r') as f:
		for line in f:
			split = line.strip().split('\t')
			# Get feature representation
			embedding_1 = get_embedding(split[0], embeddings)
			embedding_2 = get_embedding(split[1], embeddings)
			X_train.append(embedding_1 + embedding_2)
			# Get label
			label = split[2]
			Y_train.append(label)
	classes = sorted(list(set(Y_train)))
	X_train = array(X_train)
	# Convert string labels to one-hot vectors
	Y_train = label_binarize(Y_train, classes)
	Y_train = array(Y_train)
	# Split off development set from training data
	X_dev = X_train[-3066:]
	Y_dev = Y_train[-3066:]
	X_train = X_train[:-3066]
	Y_train = Y_train[:-3066]
	print(len(X_train), 'training instances')
	print(len(X_dev), 'develoment instances')
	# Test data
	X_test = []
	Y_test = []
	with open('test_data_clean.tsv', 'r') as f:
		for line in f:
			split = line.strip().split('\t')
			# Get feature representation
			embedding_1 = get_embedding(split[0], embeddings)
			embedding_2 = get_embedding(split[1], embeddings)
			X_test.append(embedding_1 + embedding_2)
	X_test = np.array(X_test)
	print(len(X_test), 'test instances')

	return X_train, X_dev, X_test, Y_train, Y_dev, classes