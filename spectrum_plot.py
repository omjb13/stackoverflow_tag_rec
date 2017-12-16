from multilabel import MultiLabelClassifier
from sys import argv
import matplotlib.pyplot as plt
import numpy as np
from helpers import create_multilabel_label_vector, get_tf_idf_matrices
from sklearn.model_selection import train_test_split
import csv
from sklearn import decomposition


def main():
    infile = argv[1]

    # INPUT PARSING
    _in = csv.reader(file(infile))
    test_size = 0.3
    shuffle_data_sets = True

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
                                                        shuffle=shuffle_data_sets)

    print "X Train and test : ", x_train.shape, x_test.shape
    print "y Train and test : ", y_train.shape, y_test.shape

    u, d, vT = np.linalg.svd(x_train)
    plt.plot(d)

    print "Showing plot..."
    plt.show()

    pca = decomposition.PCA(n_components=2)
    X_r = pca.fit(x_train).transform(x_train)

    print X_r.shape


if __name__ == '__main__':
    main()
    # data_packed = {
    #     'train': x_train,
    #     'test': x_test
    # }
    #
    # labels_packed = {
    #     'train': y_train,
    #     'test': y_test
    # }

    # mc = MultiLabelClassifier(data_packed, labels_packed)
