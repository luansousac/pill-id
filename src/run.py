# Import all dependencies
from sklearn.model_selection import train_test_split
from .classifiers import *
import numpy as np
import sklearn.metrics as metrics

# Read dataset
np.random.seed(1)
X, y = read_dataset('pill2.data')

# General settings
n_iter = 10
test_size = 0.2
labels = np.unique(y).shape[0]

# Prepare results storage
for classifier in classifiers:
    classifier['results'] = {
        'acc': np.zeros(shape=n_iter),
        'mcc': np.zeros(shape=n_iter),
        'pre': np.zeros(shape=n_iter),
        'rec': np.zeros(shape=n_iter),
        'cfm': np.zeros(shape=(n_iter, labels, labels)),
    }

# Perform experiments
for i in range(n_iter):
    # Holdout
    # Maybe experiment with another holdout, because of the imbalanced dataset
    X_train, X_test, y_train, y_test = \
        train_test_split(X, y, test_size=test_size)

    for classifier in classifiers:
        # Pipelines (training and validation, respectively)
        train_pipeline = make_pipeline(classifier)
        valid_pipeline = make_pipeline(classifier)

        # Model selection
        estimator = model_selection(valid_pipeline, classifier['hyper_params'],
                                    data=X_train, labels=y_train)

        # Train best model
        train_pipeline.fit(X_train, y_train)

        # Predict unseen data
        y_pred = estimator.predict(X_test)

        # Save results
        classifier['results']['acc'][i] = \
            metrics.accuracy_score(y_test, y_pred)
        classifier['results']['mcc'][i] = \
            metrics.matthews_corrcoef(y_test, y_pred)
        classifier['results']['pre'][i] = \
            metrics.precision_score(y_test, y_pred, average='micro')
        classifier['results']['rec'][i] = \
            metrics.recall_score(y_test, y_pred, average='micro')
        classifier['results']['cfm'][i, :, :] = \
            metrics.confusion_matrix(y_test, y_pred)
