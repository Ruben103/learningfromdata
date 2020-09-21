from pandas import read_csv

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

                # 6-class problem: books, camera, dvd, health, music, software
                labels.append(tokens[0])

        return documents, labels

    def cross_validation_split(self, data, labels):
        # data is of type ndarray.
        sets = []
        for i in range(5):
            split1 = i*(int(data.shape[0]/5))
            split2 = (i+1)*(int(data.shape[0]/5)) - 1
            sets.append([data[split1:split2], labels[split1:split2]])
        return sets

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

    def read_edible_fruit_table(self):

        table = read_csv('ediblefruit.csv')
        return table