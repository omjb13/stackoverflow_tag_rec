import csv
import math
from sys import argv

from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC

from helpers import get_tf_idf_matrices, create_multiclass_label_vector


def main():
    # CONFIG
    test_size = 0.4

    # INPUT
    infile = argv[1]
    top_100_tags_file = argv[2]
    top_100_tags = map(lambda x: x.split(',')[0].strip(),
                       open(top_100_tags_file, 'r').readlines())

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
    encoded_label_vector = create_multiclass_label_vector(labels, top_100_tags)
    print features

    x_train, x_test, y_train, y_test = train_test_split(scores,
                                                        encoded_label_vector,
                                                        test_size=test_size,
                                                        shuffle=False)

    print encoded_label_vector
    clf = OneVsRestClassifier(LinearSVC(random_state=0)).fit(x_train, y_train)
    predicted_labels = clf.predict(x_test)

    error = 0
    error_checking = zip(predicted_labels, y_test)

    # this bit is hacky, it's just to get the original question text back
    offset = int(math.floor(len(data) * (1 - test_size)))

    print "Beginning error checking..."

    for i, (p, t) in enumerate(error_checking):
        if p != t:
            current_record = data[offset + i]
            predicted, true = top_100_tags[p], top_100_tags[t]
            if predicted not in current_record[3]:
                print "Label mismatch. Predicted %s, is actually %s." % \
                      (top_100_tags[p], top_100_tags[t])
                print "Erroneous body + title : "
                print data[offset + i]
                print ""
                error += 1

    print "%d errors out of %d samples" % (error, y_test.shape[0])


if __name__ == '__main__':
    main()
