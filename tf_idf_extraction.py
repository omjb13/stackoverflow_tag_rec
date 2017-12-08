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

def get_tf_idf_matrices(data_set):
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

#read in training data
training_file = argv[1] 
_training_data = csv.reader(file(training_file))
training_data = list(_training_data)

#extract just text
just_text = [i[1] + " " + i[2] for i in training_data]

current_data_set = just_text 

#tf-idf
tokenize = lambda doc: doc.split(" ")

#check if we've received a pickled model from args
if len(argv) > 2:
    print "Loading model from %s..." % argv[1]
    with open('model.pickle','wb') as f: 
        #sklearn_tfidf = pickle.load(f)
        features = pickle.load(f)
        scores = pickle.load(f)
    print "Model loaded."
else: 
    print "Training TF-IDF model..."
    sklearn_tfidf = TfidfVectorizer(norm='l2',
        min_df=0, use_idf=True, smooth_idf=False, 
        sublinear_tf=True, tokenizer=tokenize)
    sklearn_tfidf = sklearn_tfidf.fit(current_data_set)
    features = sklearn_tfidf.get_feature_names()
    scores = sklearn_tfidf.transform(current_data_set).toarray()
    #pickle sklearn_tfidf
    #print "Done training, saving model..."
    # with open('model.pickle','wb') as f: 
    #     #pickle.dump(sklearn_tfidf, f)
    #     pickle.dump(features, f)
    #     pickle.dump(scores, f)

#print current_data_set[1]


# fs = sorted(zip(features,scores[1]), key=lambda x:x[1])
# for f,s in fs:
#     if s > 0:
#         print f, s

print "scores.shape",  scores.shape
print "features length",  len(features)

###########################

firefox_label_vector = []
all_labels = [i[3] for i in training_data]
for label_list in all_labels:
    if "firefox" in label_list.split():
        firefox_label_vector.append(1)
    else:
        firefox_label_vector.append(0)

firefox_label_vector = np.array(firefox_label_vector)
print "firefox_label_vector.shape : ", firefox_label_vector.shape
print "np.nonzero(firefox_label_vector)", np.nonzero(firefox_label_vector)

##########################

X = scores
y = firefox_label_vector
#clf = SVC(kernel='linear')
#clf = Ridge(alpha=0.001)
#clf = Lasso(alpha=10)
clf = DecisionTreeClassifier(random_state=0)

print "Training SVM now..."
clf.fit(X, y) 

#load test data
test_file = argv[1] 
_test_data = csv.reader(file(test_file))
test_data = list(test_data)
#extract just text
just_text = [i[1] + " " + i[2] for i in test_data]
features, scores = get_tf_idf_matrices(just_text)

#predict test labels
predicted_labels = clf.predict(scores)

#predicted_labels = map(lambda x:1 if x > 0 else 0, predicted_labels)

print "np.nonzero(predicted_labels)", np.nonzero(predicted_labels)

error = 0 
for p, t in zip(predicted_labels, ):
    if p != t:
        error += 1

print "%d errors out of %d samples" % (error, firefox_label_vector.shape[0])