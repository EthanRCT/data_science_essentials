"""
Random Forest Lab

Ethan Crawford
Math 403
10/24/23
"""
from platform import uname
import os
import graphviz
from uuid import uuid4
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import time

# Problem 1
class Question:
    """Questions to use in construction and display of Decision Trees.
    Attributes:
        column (int): which column of the data this question asks
        value (int/float): value the question asks about
        features (str): name of the feature asked about
    Methods:
        match: returns boolean of if a given sample answered T/F"""

    def __init__(self, column, value, feature_names):
        self.column = column
        self.value = value
        self.features = feature_names[self.column]

    def match(self, sample):
        """Returns T/F depending on how the sample answers the question
        Parameters:
            sample ((n,), ndarray): New sample to classify
        Returns:
            (bool): How the sample compares to the question"""
        return sample[self.column] >= self.value

    def __repr__(self):
        return "Is %s >= %s?" % (self.features, str(float(self.value)))

def partition(data, question):
    """Splits the data into left (true) and right (false)
    Parameters:
        data ((m,n), ndarray): data to partition
        question (Question): question to split on
    Returns:
        left ((j,n), ndarray): Portion of the data matching the question
        right ((m-j, n), ndarray): Portion of the data NOT matching the question
    """
    # Create a mask that is True where the data matches the question
    mask = data[:,question.column] >= question.value
 
    # Return the data split by the mask
    return data[mask], data[~mask]

# Helper function
def num_rows(array):
    """ Returns the number of rows in a given array """
    if array is None:
        return 0
    elif len(array.shape) == 1:
        return 1
    else:
        return array.shape[0]

# Helper function
def class_counts(data):
    """ Returns a dictionary with the number of samples under each class label
        formatted {label : number_of_samples} """
    if len(data.shape) == 1: # If there's only one row
        return {data[-1] : 1}
    counts = {}
    for label in data[:,-1]:
        if label not in counts:
            counts[label] = 0
        counts[label] += 1
    return counts

# Helper function
def info_gain(data, left, right):
    """Return the info gain of a partition of data.
    Parameters:
        data (ndarray): the unsplit data
        left (ndarray): left split of data
        right (ndarray): right split of data
    Returns:
        (float): info gain of the data"""
        
    def gini(data):
        """Return the Gini impurity of given array of data.
        Parameters:
            data (ndarray): data to examine
        Returns:
            (float): Gini impurity of the data"""
        counts = class_counts(data)
        N = num_rows(data)
        impurity = 1
        for lbl in counts:
            prob_of_lbl = counts[lbl] / N
            impurity -= prob_of_lbl**2
        return impurity
        
    p = num_rows(right)/(num_rows(left)+num_rows(right))
    return gini(data) - p*gini(right)-(1-p)*gini(left)

# Problem 2, Problem 6
def find_best_split(data, feature_names, min_samples_leaf=5, random_subset=False):
    """Find the optimal split
    Parameters:
        data (ndarray): Data in question
        feature_names (list of strings): Labels for each column of data
        min_samples_leaf (int): minimum number of samples per leaf
        random_subset (bool): for Problem 6
    Returns:
        (float): Best info gain
        (Question): Best question"""
    # Set best_gain = 0, best_question = None
    best_gain = 0
    best_question = None

    # If random_subset, select a random subset of features 
    # of size sqrt of the number of features
    if random_subset:
        num_features = len(data[0, :])-1
        random_indicies = np.random.randint(0, num_features, int(np.sqrt(num_features)))

    for column in range(len(data[0, :])-1):
        if random_subset:
            if column not in random_indicies:
                continue

        # For each column, get the unique values
        values = np.unique(data[:, column])
        
        # For each value, get the info gain
        for value in values:
            question = Question(column, value, feature_names)

            # Partition the data
            left, right = partition(data, question)

            # If the partition's size is smaller than the min_samples_leaf, skip
            if num_rows(left) < min_samples_leaf or num_rows(right) < min_samples_leaf:
                continue

            # Calculate the info gain
            gain = info_gain(data, left, right)

            # If the gain is better than the best gain, update the best gain and best question
            if gain > best_gain:
                best_gain = gain
                best_question = question
            
    return best_gain, best_question

# Problem 3
class Leaf:
    """Tree leaf node
    Attribute:
        prediction (dict): Dictionary of labels at the leaf"""
    def __init__(self,data):
        self.prediction = class_counts(data)

class Decision_Node:
    """Tree node with a question
    Attributes:
        question (Question): Question associated with node
        left (Decision_Node or Leaf): child branch
        right (Decision_Node or Leaf): child branch"""
    def __init__(self, question, left_branch, right_branch):
        self.question = question
        self.left = left_branch
        self.right = right_branch


# Prolem 4
def build_tree(data, feature_names, min_samples_leaf=5, max_depth=4, current_depth=0, random_subset=False):
    """Build a classification tree using the classes Decision_Node and Leaf
    Parameters:
        data (ndarray)
        feature_names(list or array)
        min_samples_leaf (int): minimum allowed number of samples per leaf
        max_depth (int): maximum allowed depth
        current_depth (int): depth counter
        random_subset (bool): whether or not to train on a random subset of features
    Returns:
        Decision_Node (or Leaf)"""
    # If we can't split again or depth reached, return a leaf
    if 2*min_samples_leaf > num_rows(data):
        return Leaf(data)
    
    # Find optimal split
    gain, question = find_best_split(data, feature_names, min_samples_leaf, random_subset)
    
    # If the gain is 0 or depth reached, return a leaf
    if gain == 0 or current_depth >= max_depth:
        return Leaf(data)
    
    # Partition the data
    left, right = partition(data, question)
    
    # Recursively build the branches
    left_branch = build_tree(left, feature_names, min_samples_leaf, max_depth, current_depth+1, random_subset)
    right_branch = build_tree(right, feature_names, min_samples_leaf, max_depth, current_depth+1, random_subset)
    
    # Return a Decision Node
    return Decision_Node(question, left_branch, right_branch)

# Problem 5
def predict_tree(sample, my_tree):
    """Predict the label for a sample given a pre-made decision tree
    Parameters:
        sample (ndarray): a single sample
        my_tree (Decision_Node or Leaf): a decision tree
    Returns:
        Label to be assigned to new sample"""
    # If leaf,  return the label
    # that corresponds with the most samples in the Leaf.
    if isinstance(my_tree, Leaf):
        return max(my_tree.prediction, key=my_tree.prediction.get)
    
    # Else, iterate down the tree until you reach a leaf
    else:
        if my_tree.question.match(sample):
            return predict_tree(sample, my_tree.left)
        else:
            return predict_tree(sample, my_tree.right)

def analyze_tree(dataset,my_tree):
    """Test how accurately a tree classifies a dataset
    Parameters:
        dataset (ndarray): Labeled data with the labels in the last column
        tree (Decision_Node or Leaf): a decision tree
    Returns:
        (float): Proportion of dataset classified correctly"""
    # For each sample in the dataset, predict the label
    # and compare it to the actual label
    correct = 0
    for sample in dataset:
        if sample[-1] == predict_tree(sample[:-1], my_tree):
            correct += 1
    
    # Return the proportion of correct classifications
    return correct/len(dataset)

# Problem 6
def predict_forest(sample, forest):
    """Predict the label for a new sample, given a random forest
    Parameters:
        sample (ndarray): a single sample
        forest (list): a list of decision trees
    Returns:
        Label to be assigned to new sample"""
    labels = []
    for tree in forest:
        labels.append(predict_tree(sample, tree))
    
    return max(set(labels), key=labels.count)

def analyze_forest(dataset,forest):
    """Test how accurately a forest classifies a dataset
    Parameters:
        dataset (ndarray): Labeled data with the labels in the last column
        forest (list): list of decision trees
    Returns:
        (float): Proportion of dataset classified correctly"""
    # For each sample in the dataset, predict the label
    # and compare it to the actual label
    correct = 0
    for sample in dataset:
        if sample[-1] == predict_forest(sample[:-1], forest):
            correct += 1
    
    # Return the proportion of correct classifications
    return correct/len(dataset)

# Problem 7
def prob7():
    """ Using the file parkinsons.csv, return three tuples. For tuples 1 and 2,
        randomly select 130 samples; use 100 for training and 30 for testing.
        For tuple 3, use the entire dataset with an 80-20 train-test split.
        Tuple 1:
            a) Your accuracy in a 5-tree forest with min_samples_leaf=15
                and max_depth=4
            b) The time it took to run your 5-tree forest
        Tuple 2:
            a) Scikit-Learn's accuracy in a 5-tree forest with
                min_samples_leaf=15 and max_depth=4
            b) The time it took to run that 5-tree forest
        Tuple 3:
            a) Scikit-Learn's accuracy in a forest with default parameters
            b) The time it took to run that forest with default parameters
    """
    parkinsons = np.loadtxt('parkinsons.csv', delimiter=',')[:,1:]
    features = np.loadtxt('parkinsons_features.csv', delimiter=',', dtype=str, comments=None)


    testing = 100
    training = 30

    # Tuple 1: My forest accuracy and time
    start = time.time()
    forest = []
    for _ in range(5):
        np.random.shuffle(parkinsons)
        X_train = parkinsons[:testing]
        X_test = parkinsons[testing:training+testing]
        my_tree = build_tree(X_train, features, min_samples_leaf=15, max_depth=4)
        forest.append(my_tree)
    
    my_accuracy = analyze_forest(X_test, forest)

    my_time = time.time() - start
    tuple1 = (my_accuracy, my_time)

    # Tuple 2: Scikit-Learn's accuracy and time
    start = time.time()
    forest = RandomForestClassifier(n_estimators=5, max_depth=4, min_samples_leaf=15)
    forest.fit(X_train[:,:-1], X_train[:,-1])
    sklearn_accuracy = forest.score(X_test[:,:-1], X_test[:,-1])
    sklearn_time = time.time() - start
    tuple2 = (sklearn_accuracy, sklearn_time)

    # Tuple 3: Scikit-Learn's accuracy and time on the entire dataset
    start = time.time()
    forest = RandomForestClassifier()
    train = int(len(parkinsons)*.8)
    forest.fit(parkinsons[:train,:-1], parkinsons[:train,-1])
    sklearn_accuracy = forest.score(parkinsons[train:,:-1], parkinsons[train:,-1])
    sklearn_time = time.time() - start
    tuple3 = (sklearn_accuracy, sklearn_time)

    return tuple1, tuple2, tuple3

## Code to draw a tree
def draw_node(graph, my_tree):
    """Helper function for drawTree"""
    node_id = uuid4().hex
    #If it's a leaf, draw an oval and label with the prediction
    if isinstance(my_tree, Leaf):
        graph.node(node_id, shape="oval", label="%s" % my_tree.prediction)
        return node_id
    else: #If it's not a leaf, make a question box
        graph.node(node_id, shape="box", label="%s" % my_tree.question)
        left_id = draw_node(graph, my_tree.left)
        graph.edge(node_id, left_id, label="T")
        right_id = draw_node(graph, my_tree.right)
        graph.edge(node_id, right_id, label="F")
        return node_id

def draw_tree(my_tree, filename='Digraph', leaf_class=Leaf):
    """Draws a tree"""
    # Remove the files if they already exist
    for file in [f'{filename}.gv',f'{filename}.gv.pdf']:
        if os.path.exists(file):
            os.remove(file)
    graph = graphviz.Digraph(comment="Decision Tree")
    draw_node(graph, my_tree)
    # graph.render(view=True) #This saves Digraph.gv and Digraph.gv.pdf
    in_wsl = False
    in_wsl = 'microsoft-standard' in uname().release
    if in_wsl:
        graph.render(f'{filename}.gv', view=False)
        os.system(f'cmd.exe /C start {filename}.gv.pdf')
    else:
        graph.render(view=True)

##################
# Problem 1 test #
##################
# animals = np.loadtxt('animals.csv', delimiter=',')
# features = np.loadtxt('animal_features.csv', delimiter=',', dtype=str, comments=None)
# names = np.loadtxt('animal_names.csv', delimiter=',', dtype=str)

# question = Question(column=1, value=3, feature_names=features)
# left, right = partition(animals, question)
# print(len(left), len(right))

# question = Question(column=1, value=75, feature_names=features)
# left, right = partition(animals, question)
# print(len(left), len(right))



##################
# Problem 2 test #
##################
# animals = np.loadtxt('animals.csv', delimiter=',')
# features = np.loadtxt('animal_features.csv', delimiter=',', dtype=str, comments=None)
# names = np.loadtxt('animal_names.csv', delimiter=',', dtype=str)
# print(find_best_split(animals, features))



##################
# Problem 4 test #
##################
# animals = np.loadtxt('animals.csv', delimiter=',')
# features = np.loadtxt('animal_features.csv', delimiter=',', dtype=str, comments=None)
# names = np.loadtxt('animal_names.csv', delimiter=',', dtype=str)
# my_tree = build_tree(animals, features)
# draw_tree(my_tree)



##################
# Problem 5 test #
##################
# animals = np.loadtxt('animals.csv', delimiter=',')
# features = np.loadtxt('animal_features.csv', delimiter=',', dtype=str, comments=None)
# names = np.loadtxt('animal_names.csv', delimiter=',', dtype=str)

# # Select 80 random samples for training
# np.random.shuffle(animals)
# X_train = animals[:80]
# X_test = animals[80:]

# my_tree = build_tree(X_train, features)
# print(analyze_tree(X_test, my_tree))



##################
# Problem 6 test #
##################
# animals = np.loadtxt('animals.csv', delimiter=',')
# features = np.loadtxt('animal_features.csv', delimiter=',', dtype=str, comments=None)
# names = np.loadtxt('animal_names.csv', delimiter=',', dtype=str)

# # Train forest
# forest = []
# for i in range(5):
#     np.random.shuffle(animals)
#     X_train = animals[:80]
#     X_test = animals[80:]
#     my_tree = build_tree(X_train, features, random_subset=True)
#     forest.append(my_tree)

# print(analyze_forest(X_test, forest))


##################
# Problem 7 test #
##################
# print(prob7())



