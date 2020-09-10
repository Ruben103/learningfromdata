from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from OutputService import PrintScores

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

"""
OUR COMMENT:
This part of the code reads in the trainset.txt and converts it to two lists {X,Y}
X contains sentences
Y contains the corresponding sentiment labels.
"""
def read_corpus(corpus_file, use_sentiment):
    documents = []
    labels = []
    with open(corpus_file, encoding='utf-8') as f:
        for line in f:
            tokens = line.strip().split()

            documents.append(tokens[3:])

            if use_sentiment:
                # 2-class problem: positive vs negative
                labels.append( tokens[1] )
            else:
                # 6-class problem: books, camera, dvd, health, music, software
                labels.append( tokens[0] )

    return documents, labels

def construct_classifier():
    # let's use the TF-IDF vectorizer
    tfidf = True

    # we use a dummy function as tokenizer and preprocessor,
    # since the texts are already preprocessed and tokenized.
    if tfidf:
        vec = TfidfVectorizer(preprocessor=identity,
                              tokenizer=identity)
    else:
        vec = CountVectorizer(preprocessor=identity,
                              tokenizer=identity)

    # combine the vectorizer with a Naive Bayes classifier
    classifier = Pipeline([('vec', vec),
                           ('cls', MultinomialNB())])
    return classifier
    
# a dummy function that just returns its input
def identity(x):
    return x

"""
OUR COMMENT:
A splitpoint variable is used to divide the whole dataset into 75% training and 25% test sets.
"""
X, Y = read_corpus('trainset.txt', use_sentiment=True)
split_point = int(0.75*len(X))
Xtrain = X[:split_point]
Ytrain = Y[:split_point]
Xtest = X[split_point:]
Ytest = Y[split_point:]

# let's use the TF-IDF vectorizer
tfidf = True

# we use a dummy function as tokenizer and preprocessor,
# since the texts are already preprocessed and tokenized.
if tfidf:
    vec = TfidfVectorizer(preprocessor = identity,
                          tokenizer = identity)
else:
    vec = CountVectorizer(preprocessor = identity,
                          tokenizer = identity)

# combine the vectorizer with a Naive Bayes classifier
classifier = Pipeline( [('vec', vec),
                        ('cls', MultinomialNB())] )

"""
OUR COMMENT:
The classifier object is a Pipeline object from the scikit-learn package.
From the documentation: https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html,
we observe that this object is used to apply a number of transforms using an estimator.

From the object we see that the final_estimator is a MultinomialNB, which is a Naive Bayes Estimator for multinomial
objects (the sentences have more than one words).

The fit functions fits each pattern (combination of one sentence from the set Xtrain and label from the set Ytrain)
using the MultinomialNB estimator.

There is no output to the function. However, it modifies the object classifier such that its parameters are trained
"""

# classifier.fit(Xtrain, Ytrain)

"""
OUR COMMENT:
After the Pipeline object has fitted on the training data, we call .predict to predict on the test set Xtest.
The object will perform a forward pass of the test data without updating its network paramaters.

The function takes as input a vector of input. In this case a list of sentences.

The output of the network is a vector of sentiment labels. In this case, the size of the output is 1500
"""

# Yguess = classifier.predict(Xtest)

use_sentiment = True
for i in range(2):
    classifier = construct_classifier()
    if i == 0:
        use_sentiment=False
        print("\nPrinting scores for multi-classs problem")
    else:
        use_sentiment=True
        print("\nPrinting scores for binary problem")

    X, Y = read_corpus('trainset.txt', use_sentiment=use_sentiment)
    split_point = int(0.75 * len(X))
    Xtrain = X[:split_point]
    Ytrain = Y[:split_point]
    Xtest = X[split_point:]
    Ytest = Y[split_point:]

    #let the classifier fit
    classifier.fit(Xtrain, Ytrain)
    Yguess = classifier.predict(Xtest)

    PrintScores().print_precision_score(y_test=Ytest, y_pred=Yguess)
    PrintScores().print_recall_score(y_test=Ytest, y_pred=Yguess)
    PrintScores().print_f1_score(y_test=Ytest, y_pred=Yguess)

"""
OUR COMMENT:
The last line prints the accuracy using the function accuracy_score from the sklearn.metrics package.
The accuracy is equal to the jaccard_score or hamming_score. Which, in a binary class problem is the computed
hamming distance divided by the maximum possible hamming distance.

In words, the output is equal to the perunage of correctly classified class labels.
"""
print('\naccuracy_score' ,accuracy_score(Ytest, Yguess))

print("BUGSTOPPER")