from .elm.elm import ELMClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import numpy as np

classifiers = [
    # MLP
    {
        'class': MLPClassifier,
        'params': {
            'solver': 'lbfgs',
            'activation': 'tanh',
            'alpha': 1e-5,
            'max_iter': 300,
            'tol': 1e-6
        },
        'hyper_params': {
            'clf__hidden_layer_sizes': [(n,) for n in range(10, 15)]
        }
    },
    # SVM/LIN
    {
        'class': SVC,
        'params': {
            'gamma': 'auto',
            'kernel': 'linear',
            'decision_function_shape': 'ovo'
        },
        'hyper_params': {
            'clf__C': [2 ** n for n in range(1, 6)]
        }
    },
    # SVM/RBF
    {
        'class': SVC,
        'params': {
            'gamma': 'auto',
            'kernel': 'poly',
            'decision_function_shape': 'ovo'
        },
        'hyper_params': {
            'clf__C': [2 ** n for n in range(1, 6)],
            'clf__degree': [d for d in range(2, 5)]
        }
    },
    # SVM/POL
    {
        'class': SVC,
        'params': {
            'kernel': 'rbf',
            'decision_function_shape': 'ovo'
        },
        'hyper_params': {
            'clf__C': [2 ** n for n in range(1, 6)],
            'clf__gamma': [2 ** n for n in range(-5, 6)]
        }
    },
    # KNN
    {
        'class': KNeighborsClassifier,
        'params': {},
        'hyper_params': {
            'clf__n_neighbors': [1, 2, 3]
        }
    },
    # NB
    {
        'class': GaussianNB,
        'params': {},
        'hyper_params': {}
    },
    # LDA
    {
        'class': LinearDiscriminantAnalysis,
        'params': {},
        'hyper_params': {}
    },
    # QDA
    {
        'class': QuadraticDiscriminantAnalysis,
        'params': {},
        'hyper_params': {}
    },
    # ELM
    {
        'class': ELMClassifier,
        'params': {},
        'hyper_params': {
            'clf__n_hidden': [n for n in range(15, 25)]
        }
    },
]


class NoParametersError(Exception):
    pass


def model_selection(classifier, params, data, labels, cv=5):
    search = GridSearchCV(classifier, cv=cv, param_grid=params)
    search.fit(data, labels)

    return search.best_estimator_


def read_dataset(filename):
    raw_data = np.loadtxt(filename, delimiter=',')
    features = raw_data[:, :-1]
    labels = raw_data[:, -1]

    return features, labels


def make_pipeline(classifier):
    steps = [
        ('scale', StandardScaler()),  # Z-score normalization
        ('clf', get_estimator(classifier)),  # Classifier
    ]
    pipeline = Pipeline(steps=steps)
    return pipeline


def get_estimator(classifier):
    estimator = classifier['class']
    parameters = classifier['params']
    return estimator(**parameters)
