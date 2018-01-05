import networkx as nx
import numpy as np
import scipy.io as sio
from mat2graph_measure import *
from load_data import *
import os
from sklearn_pandas import DataFrameMapper, cross_val_score
from sklearn import svm, datasets, preprocessing
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve, auc
from sklearn.utils import shuffle
from sklearn.preprocessing import normalize
import pdb

# Set data path
data_path = '/Users/michaelcraig/projects/Sedation/wavelets/wave_matrices/correlation/'

dataset = load_data(data_path)

###
#path_dir = '/Users/michaelcraig/projects/Sedation/wavelets/scripts/network_svm'
#file_name = 'dataset_corr_fisher_awake-mod.txt'
#path = os.path.join(path_dir,file_name)
#f = open(path)
#f.readline() # to skip header
#dataset = np.loadtxt(f)
###

X = dataset[:, 3:5]  # select feature columns of interest
X = normalize(X)
y = dataset[:, 2]   # select label column 0, the sedation level
# shuffle the dataset
X, y = shuffle(X, y, random_state=0)

# Create training and testing sets
# Split the dataset in two equal parts
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)

# Set the parameters by cross-validation
parameters = [{'kernel': ['rbf'],
               'gamma': [1e-4, 1e-3, 0.01, 0.1, 0.2, 0.5, 1.0],
                'C': [1, 10, 100, 1000, 10000]},
              {'kernel':['sigmoid'],
               'gamma': [1e-5, 1e-4, 1e-3, 0.01, 0.1, 0.2, 0.5, 1]}]#,
              #{'kernel':['poly'],
              # 'gamma': [1e-4, 1e-3, 0.01, 0.1, 0.2, 0.5],
              # 'C': [1, 10, 100, 1000],
              # 'degree': [1, 2, 3, 4, 5]},
             #{'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]

print("# Tuning hyper-parameters")
print(parameters)

# define svc
clf = GridSearchCV(svm.SVC(decision_function_shape='ovr'), parameters, cv=10, verbose=2)
clf.fit(X_train, y_train)

print("Best parameters set found on development set:")
print()
print(clf.best_params_)
print()
print("Grid scores on training set:")
print()
means = clf.cv_results_['mean_test_score']
stds = clf.cv_results_['std_test_score']
for mean, std, params in zip(means, stds, clf.cv_results_['params']):
    print("%0.3f (+/-%0.03f) for %r"
          % (mean, std * 2, params))

# Test classifier
print("Detailed classification report:")
print()
print("The model is trained on the full development set.")
print("The scores are computed on the full evaluation set.")
print()
y_true, y_pred = y_test, clf.predict(X_test)
print(classification_report(y_true, y_pred))
print()

print(('Test score:',clf.score(X_test,y_test)))
