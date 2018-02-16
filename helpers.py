import csv
import numpy as np

#Load data from csv file, and break it into an data list of lists (feature set) and a y list. 
def load_data(filename, split=0.8, features=[], blacklist=[], shuffle=True):
	''' load_data(filename, split=0.8, features=[], blacklist=[], shuffle=True) -> X_train, y_train, X_test, y_test
	Loads data from a csv file to four numpy arrays, X_train, y_train, X_test and y_test.
	split determines the portion of the data to be sent to the training set.
	features is a whitelist of features from the headers.
	blacklist is a blacklist of features from the headers. You cannot use both features and blacklist.
	shuffle determines if the data will be shuffled after being loaded.
	'''

	if len(features)>0 and len(blacklist)>0:
		raise ValueError("You can use one of the features or blacklist lists, but not both.")

	with open(filename, 'r') as source:
		data = []
		csv_reader = csv.reader(source)

		for row in csv_reader:
			data.append(row)	

		feature_set = data[0]
		data = data[1:]

		#Processing the blacklist or the features list.
		if features == []:
			selected_indices = [feature_set.index(k) for k in feature_set if not k in blacklist]
		else:
			selected_indices = [feature_set.index(k) for k in features]

		data = np.array(data)
		data = data[:, selected_indices]

		if shuffle:
			np.random.shuffle(data)

		training_length = int(data.shape[0] * split)
		X_train = data[:training_length-1, :-1]
		y_train = data[:training_length-1, -1]
		X_test = data[training_length:, :-1]
		y_test = data[training_length:, -1]

	return X_train, y_train, X_test, y_test

def load_test_data(filename, features=[], blacklist=[]):
	'''load_test_data(filename, features=[], blacklis[]) -> X_test, test_ids
	Special function to load test data for Analyticity 2018.
	features and blacklist are is in load_data().
	'''

	if len(features)>0 and len(blacklist)>0:
		raise ValueError("You can use one of the features or blacklist lists, but not both.")

	with open(filename, 'r') as source:
		data = []
		csv_reader = csv.reader(source)

		for row in csv_reader:
			data.append(row)	

		feature_set = data[0]
		data = data[1:]
		test_ids = np.array(data)[:, 0]

		#Processing the blacklist or the features list.
		if features == []:
			selected_indices = [feature_set.index(k) for k in feature_set if not k in blacklist]
		else:
			selected_indices = [feature_set.index(k) for k in features]

		X_test = np.array(data)
		X_test = X_test[:, selected_indices]

	return X_test, test_ids

def mean(l):
	'''mean(l) -> number
	returns the mean of a given list.
	'''
	return sum(l)/len(l)

def round(array, threshold):
	'''round(array, threshold) -> array
	rounds the numbers in a 1D array. If an element is greater than or equal to the threshold, it is rounded up to 1. Otherwise, rounds down to 0.
	Only rounds numbers between 1 and 0.
	'''

	if threshold < 0 or threshold > 1:
		raise ValueError("threshold cannot be greater than 1 or less than 0.")

	array = np.array( [ 1 if element >= threshold else 0 for element in array ] )
	return array

def normalise_data(X, normalise_indices=[], zero_to_one=True):
	'''normalise_data(X, normalise_indices=[], zero_to_one=True) -> X
	Normalises a dataset.
	normalise_indices is the list of indices to be normalised.
	zero_to_one sets whether the normalisation is on the range 0 to 1 or -1 to 1.
	'''

	# If all indices are to be normalised.
	if normalise_indices == []:
		normalise_indices = range(len(X[0]))

	for i in normalise_indices:
		nu = 0 if zero_to_one else nmean(X[:,i])
		X[:,i] = ( ( X[:,i] - nu ) / (max(X[:, i]) - min(X[:, i]) ) )

	return X