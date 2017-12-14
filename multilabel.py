import csv
import math
from sys import argv

from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import f1_score, recall_score, precision_score
import random

from helpers import get_tf_idf_matrices, create_multilabel_label_vector


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

    # MODEL TRAINING
    print "Training model now..."
    clf = OneVsRestClassifier(LinearSVC(random_state=0)).fit(x_train, y_train)
    predicted_labels = clf.predict(x_test)

    # Using the micro F1 score because we can have labels that end up with an F1
    # score of 0 as there are no samples for that label, thereby bringing down overall F1
    # Micro circumvents this by looking at global false/true positive/negatives as opposed
    # to per-class
    f1 = f1_score(y_test, predicted_labels, average="micro")
    recall = recall_score(y_test, predicted_labels, average="micro")
    precision = precision_score(y_test, predicted_labels, average="micro")

    print "Recall score is %f" % (recall)
    print "Precision score is %f" % (precision)
    print "F1 score is %f" % (f1)


if __name__ == '__main__':
    main()
