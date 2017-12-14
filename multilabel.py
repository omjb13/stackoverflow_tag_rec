import csv
from sys import argv
from sklearn.metrics import f1_score, recall_score, precision_score
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from helpers import get_tf_idf_matrices, create_multilabel_label_vector


class MultiLabelClassifier:
    def __init__(self, data, labels):
        self.x_train = data['train']
        self.x_test = data['test']
        self.y_train = labels['train']
        self.y_test = labels['test']
        self.predicted_labels = []
        self.scores = {}

    def train_classifier(self):
        self.clf = OneVsRestClassifier(LinearSVC(random_state=0)).fit(self.x_train, self.y_train)
        return self.clf

    def predict_labels(self):
        self.predicted_labels = self.clf.predict(self.x_test)
        return self.predicted_labels

    def get_accuracy_scores(self):
        # if we haven't already run prediction, predict now.
        if not self.predicted_labels:
            _ = self.predict_labels()
        # Using the micro F1 score because we can have labels that end up with an F1
        # score of 0 as there are no samples for that label, thereby bringing down overall F1
        # Micro circumvents this by looking at global false/true positive/negatives as opposed
        # to per-class
        self.scores['f1'] = f1_score(self.y_test, self.predicted_labels, average="micro")
        self.scores['recall'] = recall_score(self.y_test, self.predicted_labels, average="micro")
        self.scores['precision'] = precision_score(self.y_test, self.predicted_labels, average="micro")
        return self.scores


def main():
    # CONFIG
    test_size = 0.4

    # INPUT
    infile = argv[1]

    # INPUT PARSING
    _in = csv.reader(file(infile))
    # ignore headers
    _in.next()
    data = list(_in)

    title_and_body = [i[1] + " " + i[2] for i in data]
    labels = [i[3] for i in data]

    # FEATURE EXTRACTION 
    print "Performing TF-IDF..."

    features, scores = get_tf_idf_matrices(title_and_body)

    # we need to save the binarizer so we can convert back to a list of 
    # labels later for the error checking.

    binarizer, encoded_label_vector = create_multilabel_label_vector(labels)

    x_train, x_test, y_train, y_test = train_test_split(scores,
                                                        encoded_label_vector,
                                                        test_size=test_size,
                                                        shuffle=False)

    print "X Train and test : ", x_train.shape, x_test.shape
    print "y Train and test : ", y_train.shape, y_test.shape

    data_packed = {
        'train': x_train,
        'test': x_test
    }

    labels_packed = {
        'train': y_train,
        'test': y_test
    }

    mc = MultiLabelClassifier(data_packed, labels_packed)
    print "Training model now..."
    mc.train_classifier()
    for score_type, score in mc.get_accuracy_scores().iteritems():
        print "%s : %f" % (score_type, score)


if __name__ == '__main__':
    main()
