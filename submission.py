import numpy as np


class DecisionNode:
    """Class to represent a single node in a decision tree."""

    def __init__(self, left, right, decision_function, class_label=None):
        """Create a decision function to select between left and right nodes.
        Note: In this representation 'True' values for a decision take us to
        the left. This is arbitrary but is important for this assignment.
        Args:
            left (DecisionNode): left child node.
            right (DecisionNode): right child node.
            decision_function (func): function to decide left or right node.
            class_label (int): label for leaf node. Default is None.
        """

        self.left = left
        self.right = right
        self.decision_function = decision_function
        self.class_label = class_label

    def decide(self, feature):
        """Get a child node based on the decision function.
        Args:
            feature (list(int)): vector for feature.
        Return:
            Class label if a leaf node, otherwise a child node.
        """
        #print("feature = ", feature)
        if self.class_label is not None:
            return self.class_label

        elif self.decision_function(feature):
            return self.left.decide(feature)

        else:
            return self.right.decide(feature)
        
    


def load_csv(data_file_path, class_index=-1):
    """Load csv data in a numpy array.
    Args:
        data_file_path (str): path to data file.
        class_index (int): slice output by index.
    Returns:
        features, classes as numpy arrays if class_index is specified,
            otherwise all as nump array.
    """

    handle = open(data_file_path, 'r')
    contents = handle.read()
    handle.close()
    rows = contents.split('\n')
    out = np.array([[float(i) for i in r.split(',')] for r in rows if r])

    if(class_index == -1):
        classes= out[:,class_index]
        features = out[:,:class_index]
        return features, classes
    elif(class_index == 0):
        classes= out[:, class_index]
        features = out[:, 1:]
        return features, classes

    else:
        return out


def build_decision_tree():
    """Create a decision tree capable of handling the sample data.
    Tree is built fully starting from the root.
    Returns:
        The root node of the decision tree.
    """

    
    #start from the bottom
    #If A2 == 1 class = 1, else class = 0
    node_a2_left = DecisionNode(None, None, None, 1 )
    node_a2_right = DecisionNode(None, None, None, 0 )
    
    #If A3 == 1 class = 1, else class = 0
    node_a3_left = DecisionNode(None, None, None, 1 )
    node_a3_right = DecisionNode(None, None, None, 0 )
    
    #Use the decision to set that if A4 == 1 left node is A3
    node_a3_root = DecisionNode(None, None, lambda a: a[2] == 0)
    node_a2_root = DecisionNode(None, None, lambda a: a[1] == 0)
    
    node_a2_root.left = node_a2_left
    node_a2_root.right = node_a2_right
    
    node_a3_root.left = node_a3_left
    node_a3_root.right = node_a3_right

    node_a4_left = node_a3_root
    node_a4_right = node_a2_root
    
    
    #Use the decision to set that if A1 == 1 class = 1
    node_a1_right = DecisionNode(None, None, None, 1 )
    node_a4_root = DecisionNode(None, None, lambda a: a[3] == 0)
    node_a4_root.left = node_a4_left
    node_a4_root.right = node_a4_right
    
    node_a1_left = node_a4_root
    
    
    
    node_a1_root = DecisionNode(None, None, lambda a: a[0] == 0)
    node_a1_root.left = node_a1_left
    node_a1_root.right = node_a1_right
    
    return node_a1_root

def confusion_matrix(classifier_output, true_labels):
    """Create a confusion matrix to measure classifier performance.
    Output will in the format:
        [[true_positive, false_negative],
         [false_positive, true_negative]]
    Args:
        classifier_output (list(int)): output from classifier.
        true_labels: (list(int): correct classified labels.
    Returns:
        A two dimensional array representing the confusion matrix.
    """

    # TODO: finish this.
    tp = 0
    fn = 0
    fp = 0
    tn = 0
    #check tp
    for i in range(len(classifier_output)):
        if(classifier_output[i] == true_labels[i] and classifier_output[i] == 1):
            tp += 1
        elif(classifier_output[i] == true_labels[i] and classifier_output[i] == 0):
            tn += 1
        elif(classifier_output[i] != true_labels[i] and classifier_output[i] == 1):
            fp += 1
        elif(classifier_output[i] != true_labels[i] and classifier_output[i] == 0):
            fn += 1
    
    confusion_matrix_arr = [[tp, fn], [fp, tn]]
    
    #print("confusion matrix =" , confusion_matrix_arr)
    return confusion_matrix_arr

def precision(classifier_output, true_labels):
    """Get the precision of a classifier compared to the correct values.
    Precision is measured as:
        true_positive/ (true_positive + false_positive)
    Args:
        classifier_output (list(int)): output from classifier.
        true_labels: (list(int): correct classified labels.
    Returns:
        The precision of the classifier output.
    """
    conf_matrix = confusion_matrix(classifier_output, true_labels)
    precision_num = conf_matrix[0][0] / (conf_matrix[0][0] + conf_matrix[1][0])
    
    return precision_num

def recall(classifier_output, true_labels):
    """Get the recall of a classifier compared to the correct values.
    Recall is measured as:
        true_positive/ (true_positive + false_negative)
    Args:
        classifier_output (list(int)): output from classifier.
        true_labels: (list(int): correct classified labels.
    Returns:
        The recall of the classifier output.
    """

    conf_matrix = confusion_matrix(classifier_output, true_labels)
    recall_num = conf_matrix[0][0] / (conf_matrix[0][0] + conf_matrix[0][1])
    
    return recall_num
    

def accuracy(classifier_output, true_labels):
    """Get the accuracy of a classifier compared to the correct values.
    Accuracy is measured as:
        correct_classifications / total_number_examples
    Args:
        classifier_output (list(int)): output from classifier.
        true_labels: (list(int): correct classified labels.
    Returns:
        The accuracy of the classifier output.
    """

    conf_matrix = confusion_matrix(classifier_output, true_labels)
    correct_num = conf_matrix[0][0] + conf_matrix[1][1] 
    total = correct_num + conf_matrix[0][1] + conf_matrix[1][0] 
    
    accuracy_num = correct_num / total
    
    return accuracy_num
    

def gini_impurity(class_vector):
    """Compute the gini impurity for a list of classes.
    This is a measure of how often a randomly chosen element
    drawn from the class_vector would be incorrectly labeled
    if it was randomly labeled according to the distribution
    of the labels in the class_vector.
    It reaches its minimum at zero when all elements of class_vector
    belong to the same class.
    Args:
        class_vector (list(int)): Vector of classes given as 0 or 1.
    Returns:
        Floating point number representing the gini impurity.
    """
    np_class = np.array(class_vector)
    number, count = np.unique(np_class, return_counts = True)
    np_dict = dict(zip(number, count))
    
    total = sum(np_dict.values())
    
    for key, value in np_dict.items():
        p_i = value/total
        p_i = p_i ** 2
        np_dict[key] = p_i
    
    total_pi_2 = sum(np_dict.values())
    
    return 1 - total_pi_2
        


def gini_gain(previous_classes, current_classes):
    """Compute the gini impurity gain between the previous and current classes.
    Args:
        previous_classes (list(int)): Vector of classes given as 0 or 1.
        current_classes (list(list(int): A list of lists where each list has
            0 and 1 values).
    Returns:
        Floating point number representing the information gain.
    """
    """
    Refer - https://victorzhou.com/blog/gini-impurity/
    Gini gain = gini impurity before the split - weighetd sum of gini impurity after the split
    """
    before_split = gini_impurity(previous_classes)
    #print("before =", before_split)
    
    after_split = 0.0
    sum_items = []
    for i in range(len(current_classes)):
        after_split += len(current_classes[i]) * gini_impurity(current_classes[i])
        sum_items.append(len(current_classes[i]))
    
    weighted_sum = after_split/sum(sum_items)   
    #print("after = ", after_split)
    
    
    return before_split - weighted_sum


class DecisionTree:
    """Class for automatic tree-building and classification."""

    def __init__(self, depth_limit=float('inf')):
        """Create a decision tree with a set depth limit.
        Starts with an empty root.
        Args:
            depth_limit (float): The maximum depth to build the tree.
        """

        self.root = None
        self.depth_limit = depth_limit

    def fit(self, features, classes):
        """Build the tree from root using __build_tree__().
        Args:
            features (m x n): m examples with n features.
            classes (m x 1): Array of Classes.
        """

        self.root = self.__build_tree__(features, classes)

    def find_best_attribute(self, X,y):
        #add y to X to easilty get the split class labels
        y_new = np.array([y])
        X_new = np.concatenate((X, y_new.T), axis=1)
        gini_gain_list = []
        
        for i in range(0, (X.shape[1])): #iterate through all columns except one
            threshold = np.mean(X[:,i])
            left_set = X_new[X_new[:,i] <= threshold]
            left_class = left_set[:,-1].tolist()
            
            right_set = X_new[X_new[:,i] > threshold]
            right_class = right_set[:,-1].tolist()
            
            gain = gini_gain(y, [left_class, right_class])
            gini_gain_list.append(gain)
            
        
        #return the index of best attribute from gini_gain_list
        max_index = gini_gain_list.index(max(gini_gain_list))
        #print("best index = ", max_index )
        #print("gini gain list =", gini_gain_list)
        threshold = np.mean(X[:,max_index])
        
        left_set = X_new[X_new[:,max_index] <= threshold]
        left_class = left_set[:,-1]
        #print("left_set =", left_set.shape)
        #print("left_class =", left_class.shape)
            
        right_set = X_new[X_new[:,max_index] > threshold]
        right_class = right_set[:,-1]
        #print("right_set =", right_set.shape)
        #print("right_class =", right_class.shape)
        
        
        #return the index of best attribute from gini_gain_list and classes for left and right subsets.
        return left_set[:,0:-1], left_class, right_set[:,0:-1], right_class, max_index, threshold, sum(gini_gain_list)
            
            
    def __build_tree__(self, features, classes, depth=0):
        """Build tree that automatically finds the decision functions.
        Args:
            features (m x n): m examples with n features.
            classes (m x 1): Array of Classes.
            depth (int): depth to build tree to.
        Returns:
            Root node of decision tree.
        """
        #print("**features =", type(features), features.shape) ##np array
        #print("**classes=", type(classes), classes.shape) ##np array
        #
        
        
        #If all elements of a list are of the same class, 
        #return a leaf node with the appropriate class label.
        if(len(np.unique(classes)) == 1):
            #print("**same classes = ", classes[0])
            return DecisionNode(None, None, None, classes[0])
        
        #If a specified depth limit is reached, 
        #return a leaf labeled with the most frequent class.
        if depth >= self.depth_limit or features.shape[0] <= 3: 
            #print("**max depth reached**")
            t = classes.astype(int)
            frequent_val = np.argmax(np.bincount(t))
            return DecisionNode(None,None,None, float(frequent_val))  #change this
        
        left_features, left_classes, right_features, right_classes, alpha_best_idx, threshold, total_gain = self.find_best_attribute(features, classes)
        #print("alpha index =", alpha_best_idx)
        #print("left features =", left_features.shape)
        #print("right features =", right_features.shape)
        #threshold = np.mean(features[:, alpha_best_idx])
        
        
        if(left_features.shape[0] == 0 or right_features.shape[0] == 0): #no rows in features
            #print("**0 gain return the majority**")
            t = classes.astype(int)
            frequent_val = np.argmax(np.bincount(t))
            return DecisionNode(None,None,None, float(frequent_val)) 
            
        #DecisionNode(None, None, lambda a: a[2] == 0)
        #print("features[alpha_best_idx] <= threshold = ", feature[alpha_best_idx] <= threshold)
        dt_root = DecisionNode(None, None, lambda feature: feature[alpha_best_idx] < threshold)
        dt_root.left = self.__build_tree__(left_features, left_classes, depth + 1)
        #print("---next set---")
        dt_root.right = self.__build_tree__(right_features, right_classes, depth + 1)
              
        return dt_root
        

    def classify(self, features):
        """Use the fitted tree to classify a list of example features.
        Args:
            features (m x n): m examples with n features.
        Return:
            A list of class labels.
        """

        class_labels = []
        
        for feature in features:
            class_labels.append(self.root.decide(feature))
        
        #print(class_labels)
        return class_labels


def generate_k_folds(dataset, k):
    """Split dataset into folds.
    Randomly split data into k equal subsets.
    Fold is a tuple (training_set, test_set).
    Set is a tuple (features, classes).
    Args:
        dataset: dataset to be split.
        k (int): number of subsections to create.
    Returns:
        List of folds.
        => Each fold is a tuple of sets.
        => Each Set is a tuple of numpy arrays.
    """

    # TODO: finish this.
    features = np.array(dataset[0])
    classes = np.array([dataset[1]])
    k_folds = []
    
    combined = np.concatenate((features, classes.T), axis=1)
    
    #in place random shuffle
    np.random.shuffle(combined)
    split_arr= []
    
    len_combined = len(combined)
    
    train_len = len_combined - (len_combined//k)
    test_len = len_combined - train_len 
    
    k_folds = []
    
    for i in range(k):
        start_index = i * test_len
        end_index = start_index + test_len
        
        #print("start index = ", start_index, "end_index =", end_index)
        
        train_features = np.concatenate((combined[0:start_index, 0:-1] , combined[end_index:, 0:-1]))
        train_classes = np.concatenate((combined[0:start_index, -1] , combined[end_index:, -1]))
        test_features = combined[start_index:end_index][:, 0:-1]
        test_classes = combined[start_index:end_index][:, -1]
        
        #tr_classes = np.array([train_classes])
        #te_classes = np.array([test_classes])
        
        #train = np.concatenate((train_features, tr_classes.T), axis=1))
        #test = (np.concatenate((test_features, te_classes.T), axis=1)).tolist()
        
        train = (train_features, train_classes)
        test = (test_features, test_classes)
        #print("length train =" , len(train))
        
        k_folds.append((train,test))
    
    
    return k_folds
    

class RandomForest:
    """Random forest classification."""

    def __init__(self, num_trees, depth_limit, example_subsample_rate,
                 attr_subsample_rate):
        """Create a random forest.
         Args:
             num_trees (int): fixed number of trees.
             depth_limit (int): max depth limit of tree.
             example_subsample_rate (float): percentage of example samples.
             attr_subsample_rate (float): percentage of attribute samples.
        """

        self.trees = []
        self.num_trees = num_trees
        self.depth_limit = depth_limit
        self.example_subsample_rate = example_subsample_rate
        self.attr_subsample_rate = attr_subsample_rate

    def fit(self, features, classes):
        """Build a random forest of decision trees using Bootstrap Aggregation.
            features (m x n): m examples with n features.
            classes (m x 1): Array of Classes.
        """
        #forest = []
        y = np.array([classes])
        combined = np.concatenate((features, y.T), axis=1)
        
        
        #pick samples
        samples = int(self.example_subsample_rate * len(combined))
        attributes_ct = int(self.attr_subsample_rate * features.shape[1])
        
        #print("attributes = ", attribs)
        
        for i in range(self.num_trees):
            
            #np.random.shuffle(combined)
            #new_combined = combined[0:samples,]
            
            attribs = np.random.choice(features.shape[1], attributes_ct, replace=False)
            #print("attributes = ", attribs)
        
            new_combined = combined[np.random.choice(combined.shape[0], samples, replace=True)]
            
            #pick attributes
            """
            p = np.array([new_combined[:,attribs[0]]]).T
            q = np.array([new_combined[:,attribs[1]]]).T
            sub_features = np.hstack((p,q))
        
            for att in range(2, len(attribs)):
                t_col = np.array([new_combined[:,attribs[att]]]).T
                t_features = np.hstack((sub_features, t_col))
                sub_features = t_features
            """
            
            new_features = new_combined[:][:,attribs]
            
                
            tree = DecisionTree(depth_limit  = self.depth_limit )
            tree.fit(new_features, new_combined[:,-1])
            
            self.trees.append(tree)
        
    
    def classify(self, features):
        """Classify a list of features based on the trained random forest.
        Args:
            features (m x n): m examples with n features.
        """
        t_label = []
        for tree in self.trees:
            t_label.append(tree.classify(features))
            
        t_arr = np.array(t_label).astype(int)
        t_arr = t_arr.T   
        
        def majority(arr):
            return float(np.argmax(np.bincount(arr)))
    
        
        class_labels = np.apply_along_axis(majority, 1, t_arr)   

        return class_labels
        

class ChallengeClassifier:
    """Challenge Classifier used on Challenge Training Data."""

    def __init__(self, num_trees = 1, depth_limit = float('inf'), example_subsample_rate = 1.0,
               attr_subsample_rate = 1.0):
        """Create challenge classifier.
        Initialize whatever parameters you may need here.
        This method will be called without parameters, therefore provide
        defaults.
        """

        #self.rt = RandomForest(num_trees, depth_limit, example_subsample_rate, attr_subsample_rate)
        self.rt = DecisionTree()
    
    def fit(self, features, classes):
        """Build the underlying tree(s).
            Fit your model to the provided features.
        Args:
            features (m x n): m examples with n features.
            classes (m x 1): Array of Classes.
        """

        self.rt.fit(features, classes)

    def classify(self, features):
        """Classify a list of features.
        Classify each feature in features as either 0 or 1.
        Args:
            features (m x n): m examples with n features.
        Returns:
            A list of class labels.
        """

        return self.rt.classify(features)


class Vectorization:
    """Vectorization preparation for Assignment 5."""

    def __init__(self):
        pass

    def non_vectorized_loops(self, data):
        """Element wise array arithmetic with loops.
        This function takes one matrix, multiplies by itself and then adds to
        itself.
        Args:
            data: data to be added to array.
        Returns:
            Numpy array of data.
        """

        non_vectorized = np.zeros(data.shape)
        for row in range(data.shape[0]):
            for col in range(data.shape[1]):
                non_vectorized[row][col] = (data[row][col] * data[row][col] +
                                            data[row][col])
        return non_vectorized

    def vectorized_loops(self, data):
        """Element wise array arithmetic using vectorization.
        This function takes one matrix, multiplies by itself and then adds to
        itself.
        Args:
            data: data to be sliced and summed.
        Returns:
            Numpy array of data.
        """
        vector_1 = np.array(data)
        #vector_2 = np.array(data)
        
        vector_3 = vector_1 * vector_1
        return vector_3 + vector_1
        
        
        # TODO: finish this.
        #raise NotImplemented()

    def non_vectorized_slice(self, data):
        """Find row with max sum using loops.
        This function searches through the first 100 rows, looking for the row
        with the max sum. (ie, add all the values in that row together).
        Args:
            data: data to be added to array.
        Returns:
            Tuple (Max row sum, index of row with max sum)
        """

        max_sum = 0
        max_sum_index = 0
        for row in range(100):
            temp_sum = 0
            for col in range(data.shape[1]):
                temp_sum += data[row][col]

            if temp_sum > max_sum:
                max_sum = temp_sum
                max_sum_index = row

        return max_sum, max_sum_index

    def vectorized_slice(self, data):
        """Find row with max sum using vectorization.
        This function searches through the first 100 rows, looking for the row
        with the max sum. (ie, add all the values in that row together).
        Args:
            data: data to be sliced and summed.
        Returns:
            Tuple (Max row sum, index of row with max sum)
        """

        vectorized_1 = np.array(data)
        index = np.argmax(np.sum(vectorized_1[0:100, :], axis=1))
                
        return np.sum(vectorized_1[index]) , index

    def non_vectorized_flatten(self, data):
        """Display occurrences of positive numbers using loops.
         Flattens down data into a 1d array, then creates a dictionary of how
         often a positive number appears in the data and displays that value.
         ie, [(1203,3)] = integer 1203 appeared 3 times in data.
         Args:
            data: data to be added to array.
        Returns:
            List of occurrences [(integer, number of occurrences), ...]
        """

        unique_dict = {}
        flattened = np.hstack(data)
        for item in range(len(flattened)):
            if flattened[item] > 0:
                if flattened[item] in unique_dict:
                    unique_dict[flattened[item]] += 1
                else:
                    unique_dict[flattened[item]] = 1

        return unique_dict.items()

    def vectorized_flatten(self, data):
        """Display occurrences of positive numbers using vectorization.
         Flattens down data into a 1d array, then creates a dictionary of how
         often a positive number appears in the data and displays that value.
         ie, [(1203,3)] = integer 1203 appeared 3 times in data.
         Args:
            data: data to be added to array.
        Returns:
            List of occurrences [(integer, number of occurrences), ...]
        """

        flatten_1 = np.array(data).flatten()
        number, count = np.unique(flatten_1 [flatten_1 > 0], return_counts = True)
        my_dict = dict(zip(number, count))
        #my_list = [(number, count) for number, count in my_dict.items()] 
        return list(my_dict.items())
        

def return_your_name():
    # return your name
    # TODO: finish this
    #raise NotImplemented()
    name = "Nidhi Agrawal"
    return name
