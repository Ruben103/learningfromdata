from sklearn.tree import DecisionTreeClassifier

class ClassifierService:

    def construct_decisiontreeclassifier(self):

        classifier = DecisionTreeClassifier(criterion='entropy')
        return classifier
