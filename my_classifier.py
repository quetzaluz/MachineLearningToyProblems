class MyNaiveBayesClassifier: 
    def __init__(self):
        # An list containing the Classifier labels, EG ["miss", "match"]
        self.classifiers = {}
        # An list containing arrays (may later be converted to tuples) with feature data. Stores the Classifier label, the feature label, and the frequency within the set
        self.features = []
        self.training_set_size = 0

    def train(self, training_data):
        for data in training_data:
            self.training_set_size += 1
            if data[1] not in self.classifiers:
                self.classifiers[data[1]] = 1
            else:
                self.classifiers[data[1]] += 1
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
        all_features = len(self.features)
        classifier_freq = classifier_prob = self.classifiers.copy()
        for key in classifier_prob:
            classifier_prob[key] = float(classifier_prob[key])/float(all_features)
        # todo: assign probability value to values not present in training set. Will likely calculate an arbitrarily low value for now and persue a data Laplace Smoothing solution later.
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
