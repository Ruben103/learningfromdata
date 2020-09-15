from DataService import DataService
from ClassifierService import ClassifierService
from Calculations import Calculations

class Experiments:

    def experiment1(self, trainset, testset):
        x, y = DataService().read_corpus(trainset)
        x_train, y_train, x_test, y_test = DataService().test_train_split(x,y)

        classifier = ClassifierService().construct_decisiontreeclassifier()
        classifier.fit(x_train, y_train)

        predictions = classifier.predict(x_test)

        print("STOP")

    def experiment_edible_fruit(self):

        table = DataService().read_edible_fruit_table()
        counting_dictionary = Calculations().calculate_counting_dictionary(table)

        for key in counting_dictionary.keys():
            entropy_of_features = {}
            for feature in counting_dictionary[key].keys():
                total_observations = sum(counting_dictionary[key][feature].values())
                for item in counting_dictionary[key][feature].keys():
                    entropy_of_features[feature] = Calculations().entopy(prob=(counting_dictionary[key][feature][item]/ total_observations))



        print("")