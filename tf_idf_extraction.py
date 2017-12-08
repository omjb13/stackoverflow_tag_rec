import csv
from sklearn.feature_extraction.text import TfidfVectorizer
import string
import pickle as pickle
from sys import argv
import numpy as np
from sklearn.svm import SVC
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.tree import DecisionTreeClassifier

training_file = argv[1] 
test_file = argv[2]

def get_tf_idf_matrices(data_set):
    # expects a list of strings
    tokenize = lambda doc: doc.split(" ")
    sklearn_tfidf = TfidfVectorizer(norm='l2',
        min_df=0, use_idf=True, smooth_idf=False, 
        sublinear_tf=True, tokenizer=tokenize)
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

# read in training data
_training_data = csv.reader(file(training_file))
training_data = list(_training_data)

# extract title + body
title_and_body = [i[1] + " " + i[2] for i in training_data]

# TF IDF for training data
features, scores = get_tf_idf_matrices(title_and_body)
print "scores.shape",  scores.shape
print "features length",  len(features)

# generate label vector for firefox
all_labels = [i[3] for i in training_data]
firefox_label_vector = get_single_label_vector(all_labels, 'firefox')
print "firefox_label_vector.shape : ", firefox_label_vector.shape
print "np.nonzero(firefox_label_vector)", np.nonzero(firefox_label_vector)

# train model 
X = scores
y = firefox_label_vector
# clf = SVC(kernel='linear')
# clf = Ridge(alpha=0.001)
# clf = Lasso(alpha=10)
clf = DecisionTreeClassifier(random_state=0)

print "Training Model now..."
clf.fit(X, y) 

# load test data
_test_data = csv.reader(file(test_file))
test_data = list(_test_data)

# extract body + title 
body_and_title_test = [i[1] + " " + i[2] for i in test_data]
test_features, test_scores = get_tf_idf_matrices(body_and_title_test)

# predict test labels
predicted_labels = clf.predict(test_scores)
print "np.nonzero(predicted_labels)", np.nonzero(predicted_labels)

test_all_labels = [i[3] for i in test_data]
firefox_test_label_vector = get_single_label_vector(test_all_labels, 'firefox')

error = 0 
for p, t in zip(predicted_labels, firefox_test_label_vector):
    if p != t:
        error += 1

print "%d errors out of %d samples" % (error, firefox_test_label_vector.shape[0])