import argparse
from collections import Counter, defaultdict

import numpy
from numpy import median
from sklearn.neighbors import BallTree


class Numbers:
    """
    Class to store MNIST data
    """

    def __init__(self, location):
        import pickle, gzip

        # Load the dataset
        f = gzip.open(location, 'rb')
        train_set, valid_set, test_set = pickle.load(f)

        self.train_x, self.train_y = train_set
        self.test_x, self.test_y = valid_set
        f.close()


class Knearest:
    """
    kNN classifier
    """
    # done with the training sets
    def __init__(self, x, y, k=5):
        """
        Creates a kNN instance

        :param x: Training data input
        :param y: Training data output
        :param k: The number of nearest points to consider in classification
        """

        self._x = x
        self._kdtree = BallTree(x)
        self._y = y
        self._k = k

    def majority(self, item_indices):
        """
        Given the indices of training examples, return the majority label.  If
        there's a tie, return the median value (as implemented in numpy).

        :param item_indices: The indices of the k nearest neighbors
        """
        # checking that the array that's passed in has the number of k inputs expected
        assert len(item_indices) == self._k, "Did not get k inputs"

        # Finish this function to return the most common y value for
        # these indices
        most_common_list = Counter(item_indices).most_common()

        # get highest count found
        max_count = most_common_list[0][1]

        # check if any other value has the same count
        ties = [item for item, count in most_common_list if count == max_count]

        if len(ties) == 1:
            return ties[0]
        else:
            return median(ties)  # int() all of this?

    def classify(self, example):
        """
        Given an example, classify the example.

        :param example: A representation of an example in the same
        format as training data
        """
        # Finish this function to find the k closest points, query the
        # majority function, and return the value.

        # dist: list of distances to the neighbors of the corresponding point
        # ind: list of indices of neighbors of the corresponding point
        # example is an array, have to use brackets []
        # query() returns a 2d array with the distances already sorted
        dist, ind = self._kdtree.query([example], self._k)

        # passing in only indices to an array
        item_indices = ind[0]

        # call the majority function to find the most common value,
        # pass in the array of the indices of the nearest k points,
        # searching self._y using those specific indices
        classification = self.majority(self._y[item_indices])

        return classification

    def confusion_matrix(self, test_x, test_y):
        """
        Given a matrix of test examples and labels, compute the confusion
        matrix for the current classifier.  Should return a dictionary of
        dictionaries where d[ii][jj] is the number of times an example
        with true label ii was labeled as jj.

        :param test_x: Test data representation
        :param test_y: Test data answers
        """

        # Finish this function to build a dictionary with the
        # mislabeled examples.  You'll need to call the classify
        # function for each example.

        d = defaultdict(dict)
        data_index = 0
        # we don't give test data to the model, that's why we compare it to how it classifies x
        # iterating through examples in test data and passing them into classify
        for xx, yy in zip(test_x, test_y):
            guess = self.classify(xx)
            d[yy][guess] = d[yy].get(guess,0) + 1
            data_index += 1
            if data_index % 100 == 0:
                print("%i/%i for confusion matrix" % (data_index, len(test_x)))

        return d

    @staticmethod
    def acccuracy(confusion_matrix):
        """
        Given a confusion matrix, compute the accuracy of the underlying classifier.
        """
        total = 0  # total times something was classified
        correct = 0  # total times classification was correct

        # confusion_matrix = d as seen above
        # d[i][j], i = true label and j = how many times i was labeled as j
        # items() used to get each "true label" item (row), then the column
        for i, j in confusion_matrix.items():
            for j, count in j.items():
                total += count
                if i == j:
                    correct += count

        return float(correct) / float(total)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='KNN classifier options')
    parser.add_argument('--k', type=int, default=3,
                        help="Number of nearest points to use")
    parser.add_argument('--limit', type=int, default=1000,
                        help="Restrict training to this many examples")
    args = parser.parse_args()

    data = Numbers("mnist.pkl.gz")

    # You should not have to modify any of this code

    if args.limit > 0:
        print("Data limit: %i" % args.limit)
        knn = Knearest(data.train_x[:args.limit], data.train_y[:args.limit],
                       args.k)
    else:
        knn = Knearest(data.train_x, data.train_y, args.k)
    print("Done loading data")

    confusion = knn.confusion_matrix(data.test_x, data.test_y)
    print("\t" + "\t".join(str(x) for x in range(10)))
    print("".join(["-"] * 90))
    for ii in range(10):
        print("%i:\t" % ii + "\t".join(str(confusion[ii].get(x, 0))
                                       for x in range(10)))
    print("Accuracy: %f" % knn.acccuracy(confusion))
