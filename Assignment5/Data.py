import json
import numpy as np
from sklearn.preprocessing import label_binarize

class Data:

    def get_embedding(self,word, embeddings):
        try:
            # GloVe embeddings only have lower case words
            return embeddings[word.lower()]
        except KeyError:
            return embeddings['UNK']

    # Load noun-noun compound data
    def load_data(self):
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
                embedding_1 = self.get_embedding(split[0], embeddings)
                embedding_2 = self.get_embedding(split[1], embeddings)
                X_train.append(embedding_1 + embedding_2)
                # Get label
                label = split[2]
                Y_train.append(label)
        classes = sorted(list(set(Y_train)))
        X_train = np.array(X_train)
        # Convert string labels to one-hot vectors
        Y_train = label_binarize(Y_train, classes)
        Y_train = np.array(Y_train)
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
                embedding_1 = self.get_embedding(split[0], embeddings)
                embedding_2 = self.get_embedding(split[1], embeddings)
                X_test.append(embedding_1 + embedding_2)
        X_test = np.array(X_test)
        print(len(X_test), 'test instances')

        return X_train, X_dev, X_test, Y_train, Y_dev, classes

    # Build confusion matrix with matplotlib
    def create_confusion_matrix(self, true, pred, classes):
        import matplotlib.pyplot as plt
        from sklearn.metrics import confusion_matrix
        # Build matrix
        cm = confusion_matrix(true, pred, labels=classes)
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        # Make plot
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=90)
        plt.xlabel('Predicted label')
        plt.yticks(tick_marks, classes)
        plt.ylabel('True label')
        plt.show()
