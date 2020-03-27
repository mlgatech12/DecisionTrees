import unittest
import submission as dt
import numpy as np
import time

class DecisionTreePart1Tests(unittest.TestCase):
    """Test tree example, confusion matrix, precision, recall, and accuracy.

    Attributes:
        hand_tree (DecisionTreeNode): root node of the built example tree.
        ht_examples (list(list(int)): features for example tree.
        ht_classes (list(int)): classes for example tree."""

    def setUp(self):
        """Setup test data.
        """

        self.hand_tree = dt.build_decision_tree()
        print("tree = ", self.hand_tree)
        self.ht_examples = [[1, 0, 0, 0],
                            [1, 0, 1, 1],
                            [0, 1, 0, 0],
                            [0, 1, 1, 0],
                            [1, 1, 0, 1],
                            [0, 1, 0, 1],
                            [0, 0, 1, 1],
                            [0, 0, 1, 0]]
        self.ht_classes = [1, 1, 1, 0, 1, 0, 1, 0]

    """
    def test_hand_tree_accuracy(self):
        #Test accuracy of the tree example.
        #
        #Asserts:
        #    decide return matches true class for all classes.
        #
        for index in range(0, len(self.ht_examples)):
            decision = self.hand_tree.decide(self.ht_examples[index])
            #print("expected = ", self.ht_classes[index], "received = ", decision)

            assert decision == self.ht_classes[index]
       
    def test_confusion_matrix(self):
        #Test confusion matrix for the example tree.
        #
        #Asserts:
        #    confusion matrix is correct.
        #
        
        answer = [1, 0, 0, 1, 0, 0, 0]
        true_label = [1, 1, 1, 0, 0, 0, 0]
        test_matrix = [[1, 2], [1, 3]]

        assert np.array_equal(test_matrix, dt.confusion_matrix(answer,
                                                               true_label))
    
    def test_precision_calculation(self):
        #Test precision calculation.
        #
        #Asserts:
        #    Precision matches for all true labels.
        #
        answer = [0, 0, 0, 0, 0]
        true_label = [1, 0, 0, 0, 0]

        for index in range(0, len(answer)):
            answer[index] = 1
            precision = 1 / (1 + index)

            assert dt.precision(answer, true_label) == precision

    def test_recall_calculation(self):
        #Test recall calculation.
        #
        #Asserts:
        #    Recall matches for all true labels.
        #
        answer = [0, 0, 0, 0, 0]
        true_label = [1, 1, 1, 1, 1]
        total_count = len(answer)

        for index in range(0, len(answer)):
            answer[index] = 1
            recall = (index + 1) / ((index + 1) + (total_count - (index + 1)))

            assert dt.recall(answer, true_label) == recall

    def test_accuracy_calculation(self):
        #Test accuracy calculation.
        #
        #Asserts:
        #    Accuracy matches for all true labels.
        #
        answer = [0, 0, 0, 0, 0]
        true_label = [1, 1, 1, 1, 1]
        total_count = len(answer)

        for index in range(0, len(answer)):
            answer[index] = 1
            accuracy = dt.accuracy(answer, true_label)

            assert accuracy == ((index + 1) / total_count)
    """


class DecisionTreePart2Tests(unittest.TestCase):
    #Tests for Decision Tree Learning.
    #
    #Attributes:
    #    restaurant (dict): represents restaurant data set.
    #    dataset (data): training data used in testing.
    #    train_features: training features from dataset.
    #    train_classes: training classes from dataset.
    #
    
    def setUp(self):
        #Set up test data.
        #
        self.restaurant = {'restaurants': [0] * 6 + [1] * 6,
                           'split_patrons': [[0, 0],
                                             [1, 1, 1, 1],
                                             [1, 1, 0, 0, 0, 0]],
                           'split_food_type': [[0, 1],
                                               [0, 1],
                                               [0, 0, 1, 1],
                                               [0, 0, 1, 1]]}

        self.dataset = dt.load_csv('part23_data.csv')
        self.train_features, self.train_classes = self.dataset
        #self.train_features, self.train_classes = self.restaurant
    """
    def test_gini_impurity_max(self):
        #Test maximum gini impurity.
        #
        #Asserts:
        #    gini impurity is 0.5.
        #
        
        gini_impurity = dt.gini_impurity([1, 1, 1, 0, 0, 0])

        assert  .500 == round(gini_impurity, 3)

    def test_gini_impurity_min(self):
        #Test minimum gini impurity.
        #
        #Asserts:
        #    entropy is 0.
        #
        
        gini_impurity = dt.gini_impurity([1, 1, 1, 1, 1, 1])

        assert 0 == round(gini_impurity, 3)

    def test_gini_impurity(self):
        #Test gini impurity.
        #
        #Asserts:
        #    gini impurity is matched as expected.
        #
        
        gini_impurity = dt.gini_impurity([1, 1, 0, 0, 0, 0])

        assert round(4. / 9., 3) == round(gini_impurity, 3)
    
    
    def test_gini_gain_max(self):
        #Test maximum gini gain.
        #
        #Asserts:
        #    gini gain is 0.5.
        #
        gini_gain = dt.gini_gain([1, 1, 1, 0, 0, 0],
                                 [[1, 1, 1], [0, 0, 0]])

        assert .500 == round(gini_gain, 3)

    def test_gini_gain(self):
        #Test gini gain.
        #
        #Asserts:
        #    gini gain is within acceptable bounds
        #
        gini_gain = dt.gini_gain([1, 1, 1, 0, 0, 0],
                                 [[1, 1, 0], [1, 0, 0]])

        #print("returned gain = ", gini_gain)
        assert 0.056 == round(gini_gain, 3)
    
    def test_gini_gain_restaurant_patrons(self):
        #Test gini gain using restaurant patrons.
        #
        #Asserts:
        #    gini gain rounded to 3 decimal places matches as expected.
        #
        gain_patrons = dt.gini_gain(
            self.restaurant['restaurants'],
            self.restaurant['split_patrons'])

        assert round(gain_patrons, 3) == 0.278

    def test_gini_gain_restaurant_type(self):
        #Test gini gain using restaurant food type.
        #
        #Asserts:
        #    gini gain is 0.
        #
        gain_type = round(dt.gini_gain(
            self.restaurant['restaurants'],
            self.restaurant['split_food_type']), 2)

        assert gain_type == 0.00
    
    
    def test_decision_tree_all_data(self):
        #Test decision tree classifies all data correctly.
        #
        #Asserts:
        #    classification is 100% correct.
        #
        tree = dt.DecisionTree()
        print(self.train_features.shape, self.train_classes.shape)
        tree.fit(self.train_features, self.train_classes)
        output = tree.classify(self.train_features)

        assert (output == self.train_classes).all()
    
    def test_k_folds_test_set_count(self):
        #Test k folds returns the correct test set size.
        #
        #Asserts:
        #    test set size matches as expected.
        #
        example_count = len(self.train_features)
        k = 10
        test_set_count = example_count // k
        #print("test_set_count = " , test_set_count)
        
        ten_folds = dt.generate_k_folds(self.dataset, k)

        for fold in ten_folds:
            training_set, test_set = fold

            #print("my returned length test set = ", len(test_set[0]) )
            assert len(test_set[0]) == test_set_count

    def test_k_folds_training_set_count(self):
        #Test k folds returns the correct training set size.
        #
        #Asserts:
        #    training set size matches as expected.
        #
        example_count = len(self.train_features)
        k = 10
        training_set_count = example_count - (example_count // k)
        #print("training_set_count = " , training_set_count)
        ten_folds = dt.generate_k_folds(self.dataset, k)

        for fold in ten_folds:
            training_set, test_set = fold

            #print("my returned length train set= ", len(training_set[0]) )
            assert len(training_set[0]) == training_set_count
    
    """

class RandomForestTests(unittest.TestCase):
    #Tests for Decision Tree Learning.
    #
    #Attributes:
    #    restaurant (dict): represents restaurant data set.
    #    dataset (data): training data used in testing.
    #    train_features: training features from dataset.
    #    train_classes: training classes from dataset.
    #
    
    def setUp(self):
        #Set up test data.
        #
        #print("in setup")
        self.restaurant = {'restaurants': [0] * 6 + [1] * 6,
                           'split_patrons': [[0, 0],
                                             [1, 1, 1, 1],
                                             [1, 1, 0, 0, 0, 0]],
                           'split_food_type': [[0, 1],
                                               [0, 1],
                                               [0, 0, 1, 1],
                                               [0, 0, 1, 1]]}

        self.dataset = dt.load_csv('challenge_train.csv', class_index = 0)
        #self.dataset = dt.load_csv('part23_data.csv', class_index = -1)
        
        self.train_features, self.train_classes = self.dataset
        #print(self.train_features[0])
        #print(self.train_classes[0])
        #self.train_features, self.train_classes = self.restaurant
    """
    def test_decision_tree_all_data(self):
        #Test decision tree classifies all data correctly.
        #
        #Asserts:
        #    classification is 100% correct.
        #
        tree = dt.RandomForest(num_trees = 1, depth_limit=15, example_subsample_rate = 0.7,
                 attr_subsample_rate = 0.9)
        #print(self.train_features.shape, self.train_classes.shape)
        tree.fit(self.train_features, self.train_classes)
        output = tree.classify(self.train_features)
        
        accuracy = dt.accuracy(output, self.train_classes)
        print("accuracy = ", accuracy)
    
    
    def test_challenge_one(self):
        #Test decision tree classifies all data correctly.
        #
        #Asserts:
        #    classification is 100% correct.
        #
        
        cc = dt.ChallengeClassifier()
        
                        
        cc.fit(self.train_features, self.train_classes)
        output = cc.classify(self.train_features)
        accuracy = dt.accuracy(output, self.train_classes)
        print("accuracy = ", accuracy)
        
    """
    
    def test_challenge(self):
        #Test decision tree classifies all data correctly.
        #
        #Asserts:
        #    classification is 100% correct.
        #
        
        arr1 = [1]
        arr2 = [float('inf')]
        arr3 = [1.0]
        arr4 = [1.0]
        
        for i in arr1:
            for j in arr2:
                for k in arr3:
                    for l in arr4:
                        
                        cc = dt.ChallengeClassifier(num_trees = i, depth_limit = j, example_subsample_rate = k,
                 attr_subsample_rate = l)
        
                        #print(self.train_features.shape, self.train_classes.shape)
                        cc.fit(self.train_features, self.train_classes)
                        print("**param**")
                        print(i, ", ", j , ", ",  k ,", ",  l)
                        output = cc.classify(self.train_features)
                        accuracy = dt.accuracy(output, self.train_classes)
                        print("accuracy = ", accuracy)
        
        
        
        
        #print("output from rf = ", output)
        #assert (output == self.train_classes).all()
        
        
    
    
"""
class VectorizationWarmUpTests(unittest.TestCase):
    #Tests the Warm Up exercises for Vectorization.
    #
    #Attributes:
    #    vector (Vectorization): provides vectorization test functions.
    #    data: vectorize test data.
    #
    def setUp(self):
        #Set up test data.
        #
        self.vector = dt.Vectorization()
        self.data = dt.load_csv('vectorize.csv', 1)

    def test_vectorized_loops(self):
        #Test if vectorized arithmetic.
        #
        #Asserts:
        #    vectorized arithmetic matches looped version.
        #
        real_answer = self.vector.non_vectorized_loops(self.data)
        my_answer = self.vector.vectorized_loops(self.data)
        
        #print("real answer = ", real_answer )
        #print("my answer =", my_answer)

        assert np.array_equal(real_answer, my_answer)


    def test_vectorized_slice(self):
        #Test if vectorized slicing.
        #
        #Asserts:
        #    vectorized slicing matches looped version.
        #
        real_sum, real_sum_index = self.vector.non_vectorized_slice(self.data)
        my_sum, my_sum_index = self.vector.vectorized_slice(self.data)

        assert real_sum == my_sum
        assert real_sum_index == my_sum_index


    def test_vectorized_flatten(self):
        #Test if vectorized flattening.
        #
        #Asserts:
        #    vectorized flattening matches looped version.
        #
        answer_unique = sorted(self.vector.non_vectorized_flatten(self.data))
        my_unique = sorted(self.vector.vectorized_flatten(self.data))
        
        print("correct = " , len(answer_unique))
        print("mine = " , len(my_unique))

        assert np.array_equal(answer_unique, my_unique)


    def test_vectorized_loops_time(self):
        #Test if vectorized arithmetic speed.
        #
        #Asserts:
        #    vectorized arithmetic is faster than expected gradescope time.
        #
        start_time = time.time() * 1000
        self.vector.vectorized_loops(self.data)
        end_time = time.time() * 1000
        print("time diff loops = ", end_time - start_time)
        assert (end_time - start_time) <= 0.09

    def test_vectorized_slice_time(self):
        #Test if vectorized slicing speed.
        #
        #Asserts:
        #    vectorized slicing is faster than expected gradescope time.
        #
        start_time = time.time() * 1000
        self.vector.vectorized_slice(self.data)
        end_time = time.time() * 1000
        print("time diff slice = ", end_time - start_time)
        
        assert (end_time - start_time) <= 0.07

    def test_vectorized_flatten_time(self):
        #Test if vectorized flatten speed.
        #
        #Asserts:
        #    vectorized flatten is faster than expected gradescope time.
        start_time = time.time() * 1000
        self.vector.vectorized_flatten(self.data)
        end_time = time.time()  * 1000

        assert (end_time - start_time) <= 15.0

class NameTests(unittest.TestCase):
    def setUp(self):
        #Set up test data.
        #
        self.to_compare = "George P. Burdell"


    def test_name(self):
        #Test if vectorized arithmetic.

        #Asserts:
        #    Non Matching Name
        
        self.name = dt.return_your_name()
        assert self.name != None
        assert self.name != self.to_compare

"""

if __name__ == '__main__':
    unittest.main()
