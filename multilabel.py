import csv
import math 
import numpy as np
import pickle as pickle
from sklearn.svm import SVC
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
import string
from sys import argv

infile = argv[1] 
top_100_tags_file = argv[2]
top_100_tags = map(lambda x: x.split(',')[0].strip(),
                    open(top_100_tags_file, 'r').readlines())


def get_tf_idf_matrices(data_set):
    # expects a list of strings
    tokenize = lambda doc: map(lambda x: x.strip(), doc.split(" "))
    sklearn_tfidf = TfidfVectorizer(
        norm='l2',
        min_df=5, 
        analyzer='word',
        use_idf=True, 
        smooth_idf=False, 
        sublinear_tf=True, 
        tokenizer=tokenize,
        #max_features=50000,
        ngram_range=(1,2)
        )
    sklearn_tfidf = sklearn_tfidf.fit(data_set)
    features = sklearn_tfidf.get_feature_names()
    scores = sklearn_tfidf.transform(data_set).toarray()
    return features,scores


def get_single_label_vector(label_vector, required_label):
    single_label_vector = []
    for label_list in label_vector:
        if required_label in label_list.split():
            single_label_vector.append(1)
        else:
            single_label_vector.append(0)
    single_label_vector = np.array(single_label_vector)
    return single_label_vector


def convert_to_single_label_vector(label_vector):
    single_label_vector = []
    for label_list in label_vector:
        keep_this = label_list.split()[0]
        encoded_value = top_100_tags.index(keep_this)
        single_label_vector.append(encoded_value)
    single_label_vector = np.array(single_label_vector)
    return single_label_vector


def convert_to_multilabel_array(label_vector):
    all_labels = []
    for label_list in label_vector:
        labels = label_list.split()
        all_labels.append(labels)
    binarizer = MultiLabelBinarizer().fit(all_labels)
    final_labels = binarizer.transform(all_labels)
    return binarizer, final_labels


def main():
    # CONFIG
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
    binarizer, encoded_label_vector = convert_to_multilabel_array(labels)

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