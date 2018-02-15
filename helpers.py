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
		X_train = data[:training_length, :-1]
		y_train = data[:training_length, -1]
		X_test = data[training_length:, :-1]
		y_test = data[training_length:, -1]

	return X_train, y_train, X_test, y_test