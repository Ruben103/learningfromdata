from math import log2

class Calculations():

    def entopy(self, prob):
        # FIX
        try:
            rtp = prob * (log2(prob))
        except:
            print("")
        return rtp

    def calculate_counting_dictionary(self, table):

        counting_dictionary = {}
        for col in table.columns[:-1]:
            distinct_features = {}
            idx = 0
            for element in table[col]:
                if element not in distinct_features.keys():
                    distinct_features[element] = {'yes': 0, 'no': 0}
                else:
                    label = table['edible'].iloc[idx]
                    distinct_features[element][label] += 1
                idx+=1
            counting_dictionary[col] = distinct_features

        return counting_dictionary