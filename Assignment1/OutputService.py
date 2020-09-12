from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, accuracy_score

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
        print("\nFSCORES average=NONE:")
        for i in range(len(distinct_labels)):
            print('\t', distinct_labels[i] + ':', round(f1_score_multi[i], 4))

    def print_f1_score_macro(self, y_test, y_pred):
        distinct_labels = self.get_distinct_labels(y_test)
        f1_score_macro = f1_score(y_test, y_pred, labels=distinct_labels, average='macro')
        print("\nFSCORES average=MACRO:")

        print('\t', round(f1_score_macro, 4))

    def print_f1_score_micro(self, y_test, y_pred):
        distinct_labels = self.get_distinct_labels(y_test)
        f1_score_micro = f1_score(y_test, y_pred, labels=distinct_labels, average='micro')
        print("\nFSCORES average=MICRO:")

        print('\t',  round(f1_score_micro, 4))

    def print_confusion_matrix(self, y_test, y_pred):
        distinct_labels = self.get_distinct_labels(y_test)
        confusion_matrix_multi = confusion_matrix(y_true=y_test, y_pred=y_pred, labels=distinct_labels)
        print("\nCONFUSION MATRIX", '\nshow this in the report')

    def print_accuracy_score(self, y_test, y_pred):
        """
        OUR COMMENT:
        The last line prints the accuracy using the function accuracy_score from the sklearn.metrics package.
        The accuracy is equal to the jaccard_score or hamming_score. Which, in a binary class problem is the computed
        hamming distance divided by the maximum possible hamming distance.

        In words, the output is equal to the perunage of correctly classified class labels.
        """
        print('\naccuracy_score', accuracy_score(y_test, y_pred))

    @staticmethod
    def get_distinct_labels(items):
        distinct_labels = []
        for item in items:
            if item not in distinct_labels:
                distinct_labels.append(item)
        return distinct_labels