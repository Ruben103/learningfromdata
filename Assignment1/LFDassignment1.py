from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

from sklearn.metrics import accuracy_score

from OutputService import PrintScores
from DataService import DataService
from ClassifierService import ClassifierService

from Experiments import Experiments
"""
OUR COMMENT:
This part of the code reads in the trainset.txt and converts it to two lists {X,Y}
X contains sentences
Y contains the corresponding sentiment labels.
"""

Experiments.experiment_binary(Experiments())
Experiments.experiment_multi_class(Experiments())
