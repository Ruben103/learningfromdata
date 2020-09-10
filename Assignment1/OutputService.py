from sklearn.metrics import precision_score, recall_score, f1_score

class PrintScores:

    def print_precision_score(self, y_test, y_pred):
        distinct_labels = self.get_distinct_labels(y_test)
        precision_score_multi = precision_score(y_test, y_pred, labels=distinct_labels, average=None)
        print("PRECISION SCORES:")
        for i in range(len(distinct_labels)):
            print('\t', distinct_labels[i] + ':', round(precision_score_multi[i], 4))

    def print_recall_score(self, y_test, y_pred):
        distinct_labels = self.get_distinct_labels(y_test)
        recall_score_multi = recall_score(y_test, y_pred, labels=distinct_labels, average=None)
        print("\nRECALL SCORES:")
        for i in range(len(distinct_labels)):
            print('\t', distinct_labels[i] + ':', round(recall_score_multi[i], 4))

    def print_f1_score(self, y_test, y_pred):
        distinct_labels = self.get_distinct_labels(y_test)
        f1_score_multi = f1_score(y_test, y_pred, labels=distinct_labels, average=None)
        print("\nFSCORES:")
        # f1 = 2 * ( (precision_score[i] * recall_score[i]) / (precision_score[i] + recall_score[i]) )
        for i in range(len(distinct_labels)):
            print('\t', distinct_labels[i] + ':', round(f1_score_multi[i], 4))

    @staticmethod
    def get_distinct_labels(items):
        distinct_labels = []
        for item in items:
            if item not in distinct_labels:
                distinct_labels.append(item)
        return distinct_labels