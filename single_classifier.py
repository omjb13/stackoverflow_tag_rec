import csv
import string
import pickle as pickle
from sys import argv
import numpy as np
from sklearn.svm import SVC
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import math 

from helpers import get_tf_idf_matrices, create_single_class_label_vector

def main():
    # CONFIG
    infile = argv[1] 
    current_tag = argv[2]
    test_size = 0.4

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
    encoded_label_vector = create_single_class_label_vector(labels, current_tag)
    print features

    # TRAIN TEST SPLIT 
    # turning shuffle off so that we can cross reference back with the 
    # original non-TF-IDF'd data 
    x_train, x_test, y_train, y_test = train_test_split(scores, 
                                                        encoded_label_vector, 
                                                        test_size = test_size,
                                                        shuffle=False)

    print "X Train and test : " , x_train.shape, x_test.shape
    print "y Train and test : " , y_train.shape, y_test.shape
    print "np.nonzero(y_train) : ", np.nonzero(y_train)
    print "np.nonzero(y_test) : ", np.nonzero(y_test)

    # MODEL TRAINING 
    clf = SVC(kernel='linear')
    # clf = Ridge(alpha=0.001)
    # clf = Lasso(alpha=10)
    #clf = DecisionTreeClassifier(random_state=0)

    print "Training Model now..."
    clf.fit(x_train, y_train) 
    print "Done."

    # PREDICTION
    predicted_labels = clf.predict(x_test)
    print "np.nonzero(predicted_labels) : ", np.nonzero(predicted_labels)

    error = 0 
    error_checking = zip(predicted_labels, y_test)

    #this bit is hacky, it's just to get the original question text back
    offset = int(math.floor(len(data) * (1 - test_size)))

    print "Beginning error checking..."

    for i, (p, t) in enumerate(error_checking):
        if p != t:
            print "Label mismatch. Predicted %d, is actually %d." % (p,t)
            print "Erroneous body + title : "
            print data[offset+i]
            print ""
            error += 1

    print "%d errors out of %d samples" % (error, y_test.shape[0])


if __name__ == '__main__':
    main()