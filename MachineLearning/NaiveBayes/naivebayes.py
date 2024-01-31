"""Volume 3: Naive Bayes Classifiers."""

import numpy as np
import pandas as pd
import os
from collections import Counter
from scipy import stats
from sklearn.base import ClassifierMixin
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from scipy import stats

class NaiveBayesFilter(ClassifierMixin):
    '''
    A Naive Bayes Classifier that sorts messages into spam or ham.
    '''
    # Problem 1
    def fit(self, X, y):
        '''
        Compute the values P(C=Ham), P(C=Spam), and P(x_i|C) to fit the model.

        Parameters:
            X (pd.Series): training data
            y (pd.Series): training labels
        '''
        # Calculate P(C=Ham) and P(C=Spam)
        self.prob_ham = len(X[y == "ham"]) / len(y)
        self.prob_spam = len(X[y == "spam"]) / len(y)

        # Create a Counter object for the words in the spam messages
        ham_word_counts = Counter(" ".join(X[y == "ham"]).split())
        spam_word_counts = Counter(" ".join(X[y == "spam"]).split())

        # if there are words in the spam/ham messages that are not in the other, add them to the ham word counts with a value of 0
        for word in spam_word_counts.keys():
            if word not in ham_word_counts.keys():
                ham_word_counts[word] = 0
        
        for word in ham_word_counts.keys():
            if word not in spam_word_counts.keys():
                spam_word_counts[word] = 0

        # Calculate P(x_i|C) for each word
        ham_total_words = sum(ham_word_counts.values())
        spam_total_words = sum(spam_word_counts.values())

        # Create a dictionary of the probabilities for each word
        self.ham_probs = {word: (count+1) / (ham_total_words+2) for word, count in ham_word_counts.items()}
        self.spam_probs = {word: (count+1) / (spam_total_words+2) for word, count in spam_word_counts.items()}

        return self

    # Problem 2
    def predict_proba(self, X):
        '''
        Find ln(P(C=k,x)) for each x in X and for each class.

        Parameters:
            X (pd.Series)(N,): messages to classify

        Return:
            (ndarray)(N,2): Log probability each message is ham or spam.
                Column 0 is ham, column 1 is spam.
        '''
        def calculate_probs(X, word_probs, prior_prob):
            word_probs_arr = np.array([
                [np.log(word_probs.get(word, 1/2)) for word in word_arr]
                for word_arr in X
            ], dtype=object)
            
            # Sum the log probabilities along the second axis (axis=1)
            return np.log(prior_prob) + np.array([np.sum(message) for message in word_probs_arr])

        # Split each message into a list of words
        X = X.str.split()

        # Calculate the sum of the log probabilities for each message
        ham_probs_sum = calculate_probs(X, self.ham_probs, self.prob_ham)
        spam_probs_sum = calculate_probs(X, self.spam_probs, self.prob_spam)

        # Combine the ham and spam probabilities into a single array
        return np.vstack((ham_probs_sum, spam_probs_sum)).T
        
    # Problem 3
    def predict(self, X):
        '''
        Predict the labels of each row in X, using self.predict_proba().
        The label will be a string that is either 'spam' or 'ham'.

        Parameters:
            X (pd.Series)(N,): messages to classify

        Return:
            (ndarray)(N,): label for each message
        '''
        # Calculate the log probabilities
        log_probs = self.predict_proba(X)

        # Return the label with the highest probability
        return np.array(["ham" if ham_prob > spam_prob else "spam" for ham_prob, spam_prob in log_probs], dtype=object)

def prob4():
    """
    Create a train-test split and use it to train a NaiveBayesFilter.
    Predict the labels of the test set.
    
    Compute and return the following two values as a tuple:
     - What proportion of the spam messages in the test set were correctly identified by the classifier?
     - What proportion of the ham messages were incorrectly identified?
    """
    # Load the data
    df = pd.read_csv(os.path.join(os.getcwd(), "sms_spam_collection.csv"))
    X = df["Message"]
    y = df["Label"]

    # use train test split to split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    # train the NaiveBayesFilter
    nb = NaiveBayesFilter()
    nb.fit(X_train, y_train)

    # predict the labels of the test set
    predictions = nb.predict(X_test)

    # calculate the proportion of spam messages correctly identified
    spam_correct = np.mean(predictions[y_test == "spam"] == y_test[y_test == "spam"])

    # calculate the proportion of ham messages incorrectly identified
    ham_incorrect = np.mean(predictions[y_test == "ham"] != y_test[y_test == "ham"])

    return spam_correct, ham_incorrect

# Problem 5
class PoissonBayesFilter(ClassifierMixin):
    '''
    A Naive Bayes Classifier that sorts messages in to spam or ham.
    This classifier assumes that words are distributed like
    Poisson random variables.
    '''
    def fit(self, X, y):
        '''
        Compute the values P(C=Ham), P(C=Spam), and r_{i,k} to fit the model.

        Parameters:
            X (pd.Series): training data
            y (pd.Series): training labels
        '''
        self.ham_rates = {}
        self.spam_rates = {}
        spam = X[y == 'spam']
        ham = X[y == 'ham']

        # Calculate P(C=Ham) and P(C=Spam)
        self.prob_ham = len(ham) / len(y)
        self.prob_spam = len(spam) / len(y)

        # Calculate the number of words in the spam and ham messages
        spam_e = spam.str.split().explode()
        ham_e = ham.str.split().explode()

        # Save the size of the spam and ham messages
        self.spam_size = len(spam_e)
        self.ham_size = len(ham_e)

       # Get value counts for each class
        ham = ham_e.str.split().explode().value_counts()
        spam = spam_e.value_counts()
        
        # Save value counts for predict
        self.spam_value_counts = spam
        self.ham_value_counts = ham
        
        for word in X.str.split().explode().unique():
            # If the word is in the spam or ham index, use the formula. Else, use the formula with 0.
            if word in spam.index:
                self.spam_rates[word] = (spam.loc[word] + 1) / (self.spam_size + 2)
            else:
                self.spam_rates[word] = 1 / (self.spam_size + 2)
            
            if word in ham.index:
                self.ham_rates[word] = (ham.loc[word] + 1) / (self.ham_size + 2)
            else:
                self.ham_rates[word] = 1 / (self.ham_size + 2)     
        return self

    def predict_proba(self, X):
        '''
        Find ln(P(C=k,x)) for each x in X and for each class.

        Parameters:
            X (pd.Series)(N,): messages to classify

        Return:
            (ndarray)(N,2): Log probability each message is ham or spam.
                Column 0 is ham, column 1 is spam.
        '''
        # Create an array to store the probabilities
        probabilities = np.zeros((len(X), 2))

        # Iterate through each row
        for i, row in enumerate(X):
            row = row.split()

            # Get the unique words and their counts
            unique_words, unique_counts = np.unique(row, return_counts=True)
        
            # Get the probabilities of each value
            ham_sum=0
            spam_sum=0

            for word in unique_words:
                # Get the index of the word in the message
                word_count_msg = unique_counts[np.where(unique_words == word)[0][0]]
                
                ham_sum += stats.poisson.logpmf(word_count_msg, self.ham_rates.get(word, 1/(self.ham_size + 2))*len(row))
                spam_sum += stats.poisson.logpmf(word_count_msg, self.spam_rates.get(word, 1/(self.spam_size + 2))*len(row))

            # Set the probabilities for the message including the prior probabilities
            probabilities[i][0] = np.log(self.prob_ham) + ham_sum
            probabilities[i][1] = +np.log(self.prob_spam) + spam_sum
        
        return probabilities

    def predict(self, X):
        '''
        Predict the labels of each row in X, using self.predict_proba().
        The label will be a string that is either 'spam' or 'ham'.

        Parameters:
            X (pd.Series)(N,): messages to classify

        Return:
            (ndarray)(N,): label for each message
        '''
        maxes = np.argmax(self.predict_proba(X), axis=1)
        return np.array(['ham' if x==0 else 'spam' for x in maxes], dtype=object)


def prob6():
    """
    Create a train-test split and use it to train a PoissonBayesFilter.
    Predict the labels of the test set.
    
    Compute and return the following two values as a tuple:
     - What proportion of the spam messages in the test set were correctly identified by the classifier?
     - What proportion of the ham messages were incorrectly identified?
    """
    # Load the data
    df = pd.read_csv(os.path.join(os.getcwd(), "sms_spam_collection.csv"))
    X = df["Message"]
    y = df["Label"]

    # use train test split to split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    # train the PoissonBayesFilter
    pb = PoissonBayesFilter()

    pb.fit(X_train, y_train)

    # predict the labels of the test set
    predictions = pb.predict(X_test)

    # calculate the proportion of spam messages correctly identified
    spam_correct = np.mean(predictions[y_test == "spam"] == y_test[y_test == "spam"])

    # calculate the proportion of ham messages incorrectly identified
    ham_incorrect = np.mean(predictions[y_test == "ham"] != y_test[y_test == "ham"])

    return spam_correct, ham_incorrect
    
# Problem 7
def sklearn_naive_bayes(X_train, y_train, X_test):
    '''
    Use sklearn's methods to transform X_train and X_test, create a
    naïve Bayes filter, and classify the provided test set.

    Parameters:
        X_train (pandas.Series): messages to train on
        y_train (pandas.Series): labels for X_train
        X_test  (pandas.Series): messages to classify

    Returns:
        (ndarray): classification of X_test
    '''
    # Create a CountVectorizer
    cv = CountVectorizer()

    # Fit the CountVectorizer to the training data
    counts = cv.fit_transform(X_train)

    # Create a MultinomialNB classifier
    model = MultinomialNB()
    model.fit(counts, y_train)

    # Transform the test data
    test_counts = cv.transform(X_test)
    return model.predict(test_counts)

if __name__ == '__main__':
    df = pd.read_csv("sms_spam_collection.csv")
    X = df["Message"]
    y = df["Label"]

    #Problem 1
    nb = NaiveBayesFilter()
    nb.fit(X[:300], y[:300])

    print(nb.ham_probs['out'])
    print(nb.spam_probs['out'])

    # Problem 2
    print()
    nb = NaiveBayesFilter()
    nb.fit(X[:300], y[:300])
    print(nb.predict_proba(X[800:805]))

    # Problem 3
    nb = NaiveBayesFilter()
    nb.fit(X[:300], y[:300])
    print(nb.predict(X[800:805]))

    # Problem 4
    print()
    print(prob4())

    # Problem 5
    pb = PoissonBayesFilter()
    pb.fit(X[:300], y[:300])

    print(pb.ham_rates['in'])
    print(pb.spam_rates['in'])

    print(pb.predict_proba(X[800:805]))

    print(pb.predict(X[800:805]))
