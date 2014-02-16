# Event

# Hackbright Data Science Workshop.



# Author

# Daniel Wiesenthal.  dw@cs.stanford.edu.



# Student Contributor

# Cyd La Luz. quetzaluz@gmail.com


# What is this?

# This is a simple script illustrating the usage of the Python NLTK classifier.  It is written in Python, but the comments are intended to make it clear how to port to other languages.  The flow isn't particularly well decomposed as a program; rather, it is intended to go along linearly with the associated talk/presentation.

# The goal is to find out which chocolate a particular volunteer during the talk will like.  We have a few examples of chocolate bars that we know are either matches or not (misses), and want to use that to make a guess about an unknown bar (we don't know if it will be a match, and want to guess).

# Student Note: The challenge is to write a custom classification method based off of Naive Bayes Theorem.

# The definition of this theorem used for the purposes of this exercise, notated probabilistically, is:

# P(Fi| c, Fj) = P (Fi|c)
#               = P(C) * P(F1|c) * P(F2|c) * P(F3|c) * ...


# Further reading:

# http://www.stanford.edu/class/cs124/lec/naivebayes.pdf

# http://nltk.googlecode.com/svn/trunk/doc/book/ch06.html



# Software Setup

# For this script to work, you'll need to have Python, NLTK (a Python package), and Numpy (upon which NLTK depends) installed.  On a Mac (which all have numpy pre-installed these days), run:

# sudo easy_install pip

# sudo pip install nltk

# <cd to directory with this file>

# python classification_101.py



#Required libraries

try:

    import nltk

    from nltk.classify.util import apply_features

    import string

    import pprint

    print "Great!  Looks like you're all set re: NLTK and Python."

except Exception, e:

    print "Bummer.  Looks like you don't have NLTK and Python set up correctly.  (Exception: "+str(e)+")"

    quit()

raw_input("\n\nHit enter to get started...")

#Setting up pretty printer
pp = pprint.PrettyPrinter(indent=4)


#Some example chocolate bars.  The format is a tuple of {information about the chocolate bar} and a {value}, where "match" is a good match and "miss" is a poor/bad match.

print "Defining Training Data (hand-coded in this case, see script source)"

known_1 = ("fruity dark organic sweet chocolate", "miss")

known_2 = ("interesting spicy dark bitter", "miss")

known_3 = ("sweet caramel crunchy light salty", "match")

known_4 = ("fruity dark organic bitter", "miss")

known_5 = ("sweet dark crunchy bitter interesting fruity", "match")

known_6 = ("light milky sweet", "match")

known_7 = ("refreshing dark sweet minty", "match")

known_8 = ("dark organic bitter", "miss")

known_9 = ("dark bitter bitter plain intense ghirardelli scary", "miss")

known_10 = ("organic dark salty bitter", "miss")



known_data_points = [known_1, known_2, known_3, known_4, known_5, known_6, known_7, known_8, known_9, known_10]





raw_input("\n\nHit enter to continue...")

#Feature extractor.  Basically takes a sentence/phrase/description/whatever and and outputs a stripped version of it.  This could/should be enhanced (with, eg, stemming), as that would provide easy gains in performance, but this is a good start.  Once you get the basic flow set up for a classification project, you'll spend most of your time in feature extraction.

print "Writing Feature Extractor"

def feature_extracting_function(data_point):

    features = {} #Dictionary, roughly equivalent to a hashtable in other languages.

    data_point = ''.join(ch for ch in data_point if ch not in set(string.punctuation)) #Strip punctuation characters from the string. In Python, this happens to be usually done with a .join on the string object, but don't be thrown if you're used to other languages and this looks weird (hell, it looks weird to me), all we're doing is stripping punctuation.

    words = data_point.split() #Split data_point on whitespace, return as list

    words = [word.lower() for word in words] #Convert all words in list to lowercase.  The [] syntax is a Python "list comprehension"; Google that phrase if you're confused.



    #Create a dictionary of features (True for each feature present, implicit False for absent features).  In this case, features are words, but they could be bigger or smaller, simpler or more complex.

    for word in words:

        features["contains_word_(%s)" % word] = True

    return features





raw_input("\n\nHit enter to continue...")

print "Extracting Features from Training Set"

train_set = apply_features(feature_extracting_function, known_data_points)





raw_input("\n\nHit enter to continue...")

print "Gathering unknown data points (new data) to predict on (again, hand-coded, see script source)"

#Our query chocolate bars: we want to know whether or not they're matches

unknown_1 = "milky light sweet nutty"

unknown_2 = "dark bitter plain"

unknown_3 = "dark dark bitter beyond belief organic"

unknown_4 = "organic minty sweet dark"





raw_input("\n\nHit enter to continue...")

#Train a Naive Bayes Classifier (simple but surprisingly effective).  This isn't the only classifier one could use (dtree is another, and there are many, many more), but it's a good start.

# print "Training Naive Bayes Classifier"

# Student Note: The following Classifier function is provided by nltk. The point of this exercise is to rewrite would would be provided by nltk below.
#               My custom classifier is defined below as MyNBayesClassifier

# nb = nltk.NaiveBayesClassifier.train(train_set)

print "Training Custom Naive Bayes Classifier"

# Initial implementation of custom Bayes Classifier. 
class MyNBayesClassifier: 
    def __init__(self):
        # An list containing the Classifier labels, EG ["miss", "match"]
        self.classifiers = []
        # An list containing arrays (may later be converted to tuples) with feature data. Stores the Classifier label, the feature label, and the frequency within the set
        self.features = []
        self.training_set_size = 0

    def train(self, training_data):
        test = (1, 2, 3)
        for data in training_data:
            self.training_set_size += 1
            if data[1] not in self.classifiers:
                self.classifiers.append(data[1])
            for feature in data[0]:
                feature_exists = False
                for tup in self.features:
                    if feature == tup[1] and tup[0] == data[1]:
                        tup[2] += 1
                        feature_exists = True
                if not feature_exists:
                    # feature data:     label, feature, frequency
                    self.features.append([data[1], feature, 1])

    def classify(self, feature_list):
        #todo: refactor label probability calculation, possibly storing this as class properties
        classifier_freq = {}
        all_features = len(self.features)
        for feature in self.features:
            label = feature[0]
            try:
                classifier_freq[label] += 1
            except KeyError:
                classifier_freq[label] = 1

        classifier_prob = classifier_freq
        for key in classifier_prob:
            classifier_prob[key] = float(classifier_prob[key])/float(all_features)
        for feat in self.features:
            for feature in feature_list:
                if feature == feat[1]:
                    classifier_prob[feat[0]] += float(feat[2])/float(classifier_freq[feat[0]])
 
        # Class probabilities have been calculated: One last iteration to determine most likely
        most_likely_classifier = [0, 0]
        for key in classifier_prob:
            if classifier_prob[key] >= most_likely_classifier[1]:
                most_likely_classifier = [key, classifier_prob[key]]
        return most_likely_classifier[0]

nb = MyNBayesClassifier()
nb.train(train_set)

raw_input("\n\nHit enter to continue...")

#Make guesses about our unknown projects:

print "Predicting the class of unknown data points"

print "Prediction for unknown_1: "+str(nb.classify(feature_extracting_function(unknown_1)))

print "Prediction for unknown_2: "+str(nb.classify(feature_extracting_function(unknown_2)))

print "Prediction for unknown_3: "+str(nb.classify(feature_extracting_function(unknown_3)))

print "Prediction for unknown_4: "+str(nb.classify(feature_extracting_function(unknown_4)))





raw_input("\n\nHit enter to continue...")

#Now get some insight as to how well the classifier performs in general.  The right way to do this is to have a test set of examples that were not used to train the classifier, because otherwise you're just asking for a false sense of confidence (it will report that it does very well--well, of course!  Of course it's gonna do well on the things you trained it on--what you want to see is whether it can handle new data or not).  Read more about test, train, and validation sets to do it better.  Google "10 fold cross validation" to get started on really doing it right.

test_set = train_set #No no no no no.  Except for illustration cases like in a skillshare script, then yes. :P

print "Evaluating Accuracy on Training Set (WARNING! This is just for illustration purposes, don't use train set for evaluation in practice!)"

print "Accuracy: "+str(nltk.classify.accuracy(nb, test_set))





raw_input("\n\nHit enter to continue...")

#Print the features that are most influential in making the decision of whether it's a good match or not.  Note that many of the features are presented in a format where the feature being "None" is meaningful; this is basically meant to be read as "When contains_word_(jams) is false/none, then that matters this much..."  See the nltk page referenced above for more info.

print "Let's look deeper into the classifier..."

print str(nb.show_most_informative_features(20))





raw_input("\n\nHit enter to continue...")

#Another interesting classifier, which can print out pseudocode for making a decision (just included in one line for fun).

print "Let's explore another (not NB) classifier, Decision Tree.  Because of the inherent structure of a Decision Tree classifier, we can print it out as a series of decisions made in pseudocode."

print nltk.DecisionTreeClassifier.train(train_set).pseudocode(depth=5)

