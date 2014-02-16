# Staff Webpage Content Classifier

# This excercise applies the Natural Language Tool Kit library (NLTK) for python to classify data from 2,772 academic staff pages.
# So far this model is only classifying along the basis of two categories, 'A' and 'B', which have been sorted by instructor Daniel Wiesental (dw@cs.stanford.edu)

#Required libraries

try:

    import os

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

raw_input("\n\nHit enter to continue...")


print "Loading Training Set"

#Apply the feature_extracting_function, defined blow, to all files in the training data directories for classifier A and B

training_set_array = []
for c in ['A', 'B']:
    for f_name in os.listdir('./dataset/train/' + c):
        f = open('./dataset/train/' + c + '/' + f_name, 'r')
        training_set_array.append( (f.read(), c) )
        f.close()

testing_set_array = []
for c in ['A', 'B']:
    for f_name in os.listdir('./dataset/test/' + c):
        f = open('./dataset/test/' + c + '/' + f_name, 'r')
        testing_set_array.append( (f.read(), c) )
        f.close()

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


train_set = apply_features(feature_extracting_function, training_set_array)


raw_input("\n\nHit enter to continue...")

print "Gathering unknown data points (new data) to predict on (again, hand-coded, see script source)"




raw_input("\n\nHit enter to continue...")

#Train a Naive Bayes Classifier (simple but surprisingly effective).  This isn't the only classifier one could use (dtree is another, and there are many, many more), but it's a good start.

# print "Training Naive Bayes Classifier"

nb = nltk.NaiveBayesClassifier.train(train_set)

print "Training Custom Naive Bayes Classifier"

raw_input("\n\nHit enter to continue...")

#Make guesses about our unknown projects:

# print "Predicting the class of unknown data points"

# print "Prediction for unknown_1: "+str(nb.classify(feature_extracting_function(unknown_1)))

# print "Prediction for unknown_2: "+str(nb.classify(feature_extracting_function(unknown_2)))

# print "Prediction for unknown_3: "+str(nb.classify(feature_extracting_function(unknown_3)))

# print "Prediction for unknown_4: "+str(nb.classify(feature_extracting_function(unknown_4)))





raw_input("\n\nHit enter to continue...")

#Now get some insight as to how well the classifier performs in general.  The right way to do this is to have a test set of examples that were not used to train the classifier, because otherwise you're just asking for a false sense of confidence (it will report that it does very well--well, of course!  Of course it's gonna do well on the things you trained it on--what you want to see is whether it can handle new data or not).  Read more about test, train, and validation sets to do it better.  Google "10 fold cross validation" to get started on really doing it right.

print "Evaluating Accuracy on Training Set"

test_set = apply_features(feature_extracting_function, testing_set_array)

print "Accuracy: "+str(nltk.classify.accuracy(nb, test_set))





raw_input("\n\nHit enter to continue...")

#Print the features that are most influential in making the decision of whether it's a good match or not.  Note that many of the features are presented in a format where the feature being "None" is meaningful; this is basically meant to be read as "When contains_word_(jams) is false/none, then that matters this much..."  See the nltk page referenced above for more info.

print "Let's look deeper into the classifier..."

print str(nb.show_most_informative_features(20))





raw_input("\n\nHit enter to continue...")

#Another interesting classifier, which can print out pseudocode for making a decision (just included in one line for fun).

print "Let's explore another (not NB) classifier, Decision Tree.  Because of the inherent structure of a Decision Tree classifier, we can print it out as a series of decisions made in pseudocode."

print nltk.DecisionTreeClassifier.train(train_set).pseudocode(depth=5)

