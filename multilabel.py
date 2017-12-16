import csv
from sys import argv
from sklearn.metrics import f1_score, recall_score, precision_score
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
import numpy as np
from helpers import get_tf_idf_matrices, create_multilabel_label_vector
import math


class MultiLabelClassifier:
    def __init__(self, data, labels):
        self.x_train = data['train']
        self.x_test = data['test']
        self.y_train = labels['train']
        self.y_test = labels['test']
        # TODO - don't initialize a numpy array, that's silly.
        self.predicted_labels = np.array([])
        self.scores = {}

    def train_classifier(self, method='svm', use='original'):
        if use == 'pca':
            data_set = self.reduced_x_train
        else:
            data_set = self.x_train
        if method == 'linear_svm':
            self.clf = OneVsRestClassifier(LinearSVC(random_state=0)).fit(data_set, self.y_train)
        elif method == 'dtree':
            self.clf = OneVsRestClassifier(DecisionTreeClassifier()).fit(data_set, self.y_train)
        elif method == 'sgd':
            #uses hinge loss by default
            self.clf = OneVsRestClassifier(SGDClassifier(n_jobs=-1)).fit(data_set, self.y_train)
        elif method == 'svm':
            self.clf = OneVsRestClassifier(SVC()).fit(data_set, self.y_train)
        return self.clf

    def predict_labels(self):
        self.predicted_labels = self.clf.predict(self.x_test)
        return self.predicted_labels

    def get_accuracy_scores(self):
        # if we haven't already run prediction, predict now.
        if self.predicted_labels.shape == 0:
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
    verbose_error_checking = False

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

    features, scores = get_tf_idf_matrices(title_and_body, max_features=20000)

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
    predicted_labels = mc.predict_labels()
    for score_type, score in mc.get_accuracy_scores().iteritems():
        print "%s : %f" % (score_type, score)

    # verbose_error_checking
    if verbose_error_checking:
        # convert back to lists of labels
        predicted_labels = binarizer.inverse_transform(predicted_labels)
        y_test = binarizer.inverse_transform(y_test)
        error = 0
        error_checking = zip(predicted_labels, y_test)
        # this bit is hacky, it's just to get the original question text back
        offset = int(math.floor(len(data) * (1 - test_size)))
        print "Beginning error checking..."
        for i, (p, t) in enumerate(error_checking):
            p = set(p)
            t = set(t)
            # do any of the predicted labels overlap with the true labels?
            # if so, not an error.
            if not p.intersection(t):
                print "predicted : ", p
                print "true : ", t
                current_record = data[offset + i]
                print "entire record : \n", current_record
                error += 1
                print ""
        print "%d errors out of %d samples" % (error, len(y_test))


if __name__ == '__main__':
    main()
