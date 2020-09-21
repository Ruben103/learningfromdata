from math import log2

class Calculations():

    def entopy(self, prob):
        # FIX
        if prob == 0.0:
            rtp = - 0.0
        else:
            rtp = - (prob * (log2(prob)))
        return rtp

    def calculate_counting_dictionary(self, table):

        counting_dictionary = {}
        for col in table.columns[:-1]:
            distinct_features = {}
            idx = 0
            for element in table[col]:
                if element not in distinct_features.keys():
                    distinct_features[element] = {'yes': 0, 'no': 0}
                    label = table['edible'].iloc[idx]
                    distinct_features[element][label] += 1
                else:
                    label = table['edible'].iloc[idx]
                    distinct_features[element][label] += 1
                idx+=1
            counting_dictionary[col] = distinct_features

        return counting_dictionary

    def return_total_observations(self, dict, feature):

        return sum(dict[feature].values())