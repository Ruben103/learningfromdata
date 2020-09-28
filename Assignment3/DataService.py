from pandas import read_csv
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

from numpy import concatenate, array, asarray


class DataService:

    def read_corpus(self, corpus_file):
        """
        OUR COMMENT:
        The function read_corpus  reads in the trainset.txt and converts it to two lists {X,Y}
        X contains sentences
        Y contains the corresponding sentiment labels.
        """
        documents = []
        labels = []
        with open(corpus_file, encoding='utf-8') as f:
            for line in f:
                tokens = line.strip().split()

                documents.append(tokens[3:])

                # binary problem
                labels.append(tokens[1])

        return documents, labels

    def cross_validation_split(self, data, labels):
        # data is of type ndarray.
        sets = []
        for i in range(5):
            split1 = i*(int(data.shape[0]/5))
            split2 = (i+1)*(int(data.shape[0]/5)) - 1
            sets.append([ data[split1:split2], labels[split1:split2] ])
        return sets

    def get_distinct_labels(self, labels):
        distinct_labels = []
        for label in labels:
            if label not in distinct_labels:
                distinct_labels.append(label)
        return distinct_labels

    def labels_string_to_float(self, labels):
        distinct_labels = self.get_distinct_labels(labels)
        conversion_dict = {}
        count = 0
        for label in labels:
            labels[count] = distinct_labels.index(label)
            conversion_dict[distinct_labels.index(label)] = label
            count+=1
        return conversion_dict, labels

    def construct_union_set(self, set, sets):
        union_set = None
        for elem in sets:
            if elem[1] != set[1]:
                if union_set is not None:
                    union_set[0] = concatenate([union_set[0].copy(), elem[0].copy()])
                    union_set[1] += elem[1].copy()
                else:
                    union_set = [elem[0].copy(), elem[1].copy()]
        return union_set

    def test_train_split(self, x, y):
        """
        OUR COMMENT:
        A splitpoint variable is used to divide the whole dataset into 75% training and 25% test sets.
        """

        split_point1 = int(0.8 * len(x))
        x_train = x[:split_point1]
        y_train = y[:split_point1]
        x_test = x[split_point1:]
        y_test = y[split_point1:]

        return x_train, y_train, x_test, y_test

    def test_dev_train_split(self, x, y):
        """
        OUR COMMENT:
        A splitpoint variable is used to divide the whole dataset into 75% training and 25% test sets.
        """

        split_point1 = int(0.6 * len(x))
        split_point2 = int(0.8 * len(x))
        x_train = x[:split_point1]
        y_train = y[:split_point1]
        x_dev = x[split_point1:split_point2]
        y_dev = y[split_point1:split_point2]
        x_test = x[split_point2:]
        y_test = y[split_point2:]

        return x_train, y_train, x_dev, y_dev, x_test, y_test

    def read_edible_fruit_table(self):

        table = read_csv('ediblefruit.csv')
        return table

    def vectorize_input(self, x):
        def identity(x):
            return x

        vec = TfidfVectorizer(preprocessor=identity, tokenizer=identity)
        x_vec = vec.fit_transform(x)

        return x_vec.toarray()

    def get_features_from_data(self, documents, words):

        for document in documents:
            documents[documents.index(document)] = [word for word in document if word in words]
        return documents