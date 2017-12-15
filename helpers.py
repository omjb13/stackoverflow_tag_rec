import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer


def get_tf_idf_matrices(data_set, min_df=5, ngram_range=(1,2), max_features=None):
    # expects a list of strings
    def tokenize(x):
        return x.strip().split(" ")
    sklearn_tfidf = TfidfVectorizer(
        norm='l2',
        min_df=min_df,
        analyzer='word',
        use_idf=True,
        smooth_idf=False,
        sublinear_tf=True,
        tokenizer=tokenize,
        max_features=max_features,
        ngram_range=ngram_range
    )
    sklearn_tfidf = sklearn_tfidf.fit(data_set)
    features = sklearn_tfidf.get_feature_names()
    scores = sklearn_tfidf.transform(data_set).toarray()
    return features, scores


def create_single_class_label_vector(label_vector, required_label):
    # returns a one-hot encoded vector based on whether current_label
    # is present in each row of label_vector or not.
    single_label_vector = []
    for label_list in label_vector:
        if required_label in label_list.split():
            single_label_vector.append(1)
        else:
            single_label_vector.append(0)
    single_label_vector = np.array(single_label_vector)
    return single_label_vector


def create_multiclass_label_vector(label_vector, list_of_labels):
    # this class picks the 1st label for each row of label_vector, 
    # then put it's index in list_of_labels into the return vector.
    single_label_vector = []
    for label_list in label_vector:
        keep_this = label_list.split()[0]
        encoded_value = list_of_labels.index(keep_this)
        single_label_vector.append(encoded_value)
    single_label_vector = np.array(single_label_vector)
    return single_label_vector


def create_multilabel_label_vector(label_vector):
    all_labels = []
    for label_list in label_vector:
        labels = label_list.split()
        all_labels.append(labels)
    binarizer = MultiLabelBinarizer().fit(all_labels)
    final_labels = binarizer.transform(all_labels)
    return binarizer, final_labels
