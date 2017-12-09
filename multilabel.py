import csv
import math 
import numpy as np
import pickle as pickle
from sklearn.svm import SVC
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import train_test_split
import string
from sys import argv

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
                                                        test_size = test_size,
                                                        shuffle=False)

    print "X Train and test : " , x_train.shape, x_test.shape
    print "y Train and test : " , y_train.shape, y_test.shape

    clf = OneVsRestClassifier(LinearSVC(random_state=0)).fit(x_train, y_train)
    predicted_labels = clf.predict(x_test)

    #convert back to lists of labels
    predicted_labels = binarizer.inverse_transform(predicted_labels)
    y_test = binarizer.inverse_transform(y_test)

    error = 0 
    error_checking = zip(predicted_labels, y_test)

    #this bit is hacky, it's just to get the original question text back
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
            current_record = data[offset+i]
            print "entire record : \n", current_record
            error += 1
            print ""
    print "%d errors out of %d samples" % (error, len(y_test))
    

if __name__ == '__main__':
    main()