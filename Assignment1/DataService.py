
class DataService:

    def read_corpus(self, corpus_file, use_sentiment):
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

                if use_sentiment:
                    # 2-class problem: positive vs negative
                    labels.append(tokens[1])
                else:
                    # 6-class problem: books, camera, dvd, health, music, software
                    labels.append(tokens[0])

        return documents, labels

    def test_train_split(self, x, y):
        """
        OUR COMMENT:
        A splitpoint variable is used to divide the whole dataset into 75% training and 25% test sets.
        """

        split_point = int(0.75 * len(x))
        x_train = x[:split_point]
        y_train = y[:split_point]
        x_test = x[split_point:]
        y_test = y[split_point:]

        return x_train, y_train, x_test, y_test